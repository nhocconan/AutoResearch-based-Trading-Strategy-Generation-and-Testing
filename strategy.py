#!/usr/bin/env python3
# 6h_ParabolicSAR_1dTrend_Volume
# Hypothesis: Uses Parabolic SAR for trend-following entries on 6h timeframe, filtered by daily trend structure and volume spikes.
# Long when: daily uptrend (HH & HL), volume > 1.5x 20-period average, and Parabolic SAR flips below price (bullish signal).
# Short when: daily downtrend (LH & LL), volume > 1.5x 20-period average, and Parabolic SAR flips above price (bearish signal).
# Exit when Parabolic SAR reverses position or daily trend breaks.
# Designed to capture strong trends while avoiding false breakouts in low-volume conditions.
# Works in bull markets by catching uptrends early and in bear markets by catching downtrends.
# Parabolic SAR provides clear entry/exit signals with built-in acceleration factor.

name = "6h_ParabolicSAR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for trend structure (HH, HL, LH, LL)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Higher High: today's high > yesterday's high
    hh = high_1d > np.roll(high_1d, 1)
    # Higher Low: today's low > yesterday's low
    hl = low_1d > np.roll(low_1d, 1)
    # Lower High: today's high < yesterday's high
    lh = high_1d < np.roll(high_1d, 1)
    # Lower Low: today's low < yesterday's low
    ll = low_1d < np.roll(low_1d, 1)
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    # First bar: no previous day, set to False
    uptrend[0] = False
    downtrend[0] = False
    
    # --- Parabolic SAR calculation ---
    # Initialize SAR with first high-low range
    sar = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = high[0]  # extreme point
    
    sar[0] = low[0]  # start with SAR at low
    
    for i in range(1, n):
        # Prior SAR
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        
        # Check for trend reversal
        if trend[i-1] == 1:  # was in uptrend
            if low[i] <= sar[i]:  # price broke below SAR -> downtrend
                trend[i] = -1
                sar[i] = ep  # SAR becomes prior EP
                ep = low[i]  # reset EP to current low
                af = 0.02  # reset acceleration factor
            else:  # still in uptrend
                trend[i] = 1
                if high[i] > ep:  # new high
                    ep = high[i]
                    af = min(af + 0.02, max_af)
                # SAR should not exceed prior two lows
                sar[i] = min(sar[i], low[i-1], low[i-2] if i>=2 else low[i-1])
        else:  # was in downtrend
            if high[i] >= sar[i]:  # price broke above SAR -> uptrend
                trend[i] = 1
                sar[i] = ep  # SAR becomes prior EP
                ep = high[i]  # reset EP to current high
                af = 0.02  # reset acceleration factor
            else:  # still in downtrend
                trend[i] = -1
                if low[i] < ep:  # new low
                    ep = low[i]
                    af = min(af + 0.02, max_af)
                # SAR should not be below prior two highs
                sar[i] = max(sar[i], high[i-1], high[i-2] if i>=2 else high[i-1])
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d trend indicators to 6h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Parabolic SAR (needs at least 2 periods) and volume MA(20)
    start_idx = max(2, 20)  # SAR needs 2, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sar[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1d
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: daily uptrend + volume spike + price above SAR (bullish SAR)
                if close[i] > sar[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: daily downtrend + volume spike + price below SAR (bearish SAR)
                if close[i] < sar[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price falls below SAR OR daily uptrend breaks
                if close[i] < sar[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above SAR OR daily downtrend breaks
                if close[i] > sar[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals