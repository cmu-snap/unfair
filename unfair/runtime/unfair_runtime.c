// Intercepts incoming IPv4/TCP packets, extracts useful header fields and
// metrics, and passes them to userspace.
//
// Based on tcprtt.py and tcpdrop.py:
//     https://github.com/iovisor/bcc/blob/master/tools/tcprtt.py
//     https://github.com/iovisor/bcc/blob/master/tools/tcpdrop.py

#include <bcc/proto.h>
#include <linux/ip.h>
#include <linux/ktime.h>
#include <linux/skbuff.h>
#include <linux/tcp.h>
#include <linux/time.h>
#include <net/ip.h>
#include <net/sock.h>

struct pkt_t
{
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u32 seq;
    u32 srtt_us;
    u32 tsval;
    u32 tsecr;
    u32 total_bytes;
    u32 ihl_bytes;
    u32 thl_bytes;
    u32 payload_bytes;
    // Required for time_us to be 64 bits.
    u32 padding;
    u64 time_us;
};

BPF_PERF_OUTPUT(pkts);

// Need to redefine these because the BCC rewriter does not support rewriting
// ip_hdr()'s the internal dereferences of skb members.
// Based on: https://github.com/iovisor/bcc/blob/master/tools/tcpdrop.py
static inline struct iphdr *skb_to_iphdr(const struct sk_buff *skb)
{
    // unstable API. verify logic in ip_hdr() -> skb_network_header().
    return (struct iphdr *)(skb->head + skb->network_header);
}

// Need to redefine these because the BCC rewriter does not support rewriting
// tcp_hdr()'s the internal dereferences of skb members.
// Based on: https://github.com/iovisor/bcc/blob/master/tools/tcpdrop.py
static struct tcphdr *skb_to_tcphdr(const struct sk_buff *skb)
{
    // unstable API. verify logic in tcp_hdr() -> skb_transport_header().
    return (struct tcphdr *)(skb->head + skb->transport_header);
}

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb)
{
    if (skb == NULL)
    {
        return 0;
    }
    // Check this is IPv4.
    if (skb->protocol != htons(ETH_P_IP))
    {
        return 0;
    }

    struct iphdr *ip = skb_to_iphdr(skb);
    // Check this is TCP.
    if (ip->protocol != IPPROTO_TCP)
    {
        return 0;
    }
    struct pkt_t pkt = {};
    pkt.saddr = ip->saddr;
    pkt.daddr = ip->daddr;

    struct tcphdr *tcp = skb_to_tcphdr(skb);
    u16 sport = tcp->source;
    u16 dport = tcp->dest;
    pkt.dport = ntohs(dport);
    pkt.sport = ntohs(sport);
    pkt.seq = tcp->seq;

    struct tcp_sock *ts = tcp_sk(sk);
    pkt.srtt_us = ts->srtt_us >> 3;
    // TODO: For the timestamp option, we also need to parse the sent packets.
    // We use the timestamp option to determine the RTT. But what if we just
    // use srtt instead? Let's start with that.
    pkt.tsval = ts->rx_opt.rcv_tsval;
    pkt.tsecr = ts->rx_opt.rcv_tsecr;

    // Determine the total size of the IP packet.
    u16 total_B = ip->tot_len;
    pkt.total_B = ntohs(total_B);

    // Determine the size of the IP header. The header length is in a bitfield,
    // but BPF cannot read bitfield elements. So we need to read a larger chunk
    // of bytes and extract the header length from that. Same for the TCP
    // header. We only read a single byte, so we do not need to use ntohs().
    u8 ihl;
    // The IP header length is the first field in the IP header.
    bpf_probe_read(&ihl, sizeof(ihl), &ip->tos - 1);
#if __BYTE_ORDER == __LITTLE_ENDIAN
    ihl = (ihl & 0xf0) >> 4;
#elif __BYTE_ORDER == __BIG_ENDIAN
    ihl = ihl & 0x0f;
#endif
    pkt.ihl_B = (u32)ihl * 4;

    // Determine the size of the TCP header. See notes for IP header length.
    u8 thl;
    // The TCP data offset is located after the ACK sequence number in the TCP
    // header.
    bpf_probe_read(&thl, sizeof(thl), &tcp->ack_seq + 4);
#if __BYTE_ORDER == __LITTLE_ENDIAN
    thl = (thl & 0x0f) >> 4;
#elif __BYTE_ORDER == __BIG_ENDIAN
    thl = (thl & 0xf0) >> 4;
#endif
    pkt.thl_B = (u32)thl * 4;

    // The TCP payload is the total IP packet length minus IP header minus TCP
    // header.
    pkt.payload_B = pkt.total_B - pkt.ihl_B - pkt.thl_B;

    // BPF has trouble extracting the time the proper way
    // (skb_get_timestamp()), so we do this manually. The skb's raw timestamp
    // is just a u64 in nanoseconds.
    ktime_t tstamp = skb->tstamp;
    pkt.time_us = (u64)tstamp / 1000000;
    pkts.perf_submit(ctx, &pkt, sizeof(pkt));
    return 0;
}