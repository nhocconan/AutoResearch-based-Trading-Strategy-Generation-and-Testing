#!/usr/bin/env python3
"""
1h_4h_1d_vwap_pullback_v1
Hypothesis: Use 4h VWAP trend for direction, 1d VWAP for value area, and 1h for entry timing.
- Trend: Price above/below 4h VWAP (institutional fair value)
- Value: Pullback to 1d VWAP (daily equilibrium)
- Entry: Price crosses VWAP in direction of trend on 1h
- Exit: Opposite VWAP cross or trend reversal
- Session filter: 08-20 UTC to avoid low-volume periods
- Target: 20-40 trades/year to stay within fee limits
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_vwap_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # 4h VWAP for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    tpv_4h = typical_price_4h * volume_4h
    
    # Cumulative VWAP calculation for 4h
    cum_tpv_4h = np.cumsum(tpv_4h)
    cum_vol_4h = np.cumsum(volume_4h)
    vwap_4h = np.divide(cum_tpv_4h, cum_vol_4h, out=np.full_like(cum_tpv_4h, np.nan), where=cum_vol_4h!=0)
    
    # Reset VWAP at each 4h bar start (simplified: use last value of each bar)
    vwap_4h_reset = np.full(len(vwap_4h), np.nan)
    for i in range(1, len(vwap_4h)):
        if i % 1 == 0:  # Each 4h bar, reset cumulative
            vwap_4h_reset[i] = vwap_4h[i]
        else:
            vwap_4h_reset[i] = vwap_4h_reset[i-1]
    
    # Actually, simpler: use typical price average for trend
    # Trend: close vs 4h typical price average (more stable)
    tp_4h = typical_price_4h
    vwap_4h_trend = np.full(len(tp_4h), np.nan)
    for i in range(len(tp_4h)):
        start = max(0, i-19)  # 20-period average
        vwap_4h_trend[i] = np.mean(tp_4h[start:i+1])
    
    # 1d VWAP for value area (daily equilibrium)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    tpv_1d = typical_price_1d * volume_1d
    
    # Daily VWAP (reset each day)
    vwap_1d = np.full(len(typical_price_1d), np.nan)
    for i in range(len(typical_price_1d)):
        start = max(0, i)  # Daily reset - use only current day's data
        if i == 0 or i > 0 and df_1d.index[i].date() != df_1d.index[i-1].date():
            # New day, reset
            cum_sum = tpv_1d[i]
            vol_sum = volume_1d[i]
        else:
            cum_sum += tpv_1d[i]
            vol_sum += volume_1d[i]
        if vol_sum > 0:
            vwap_1d[i] = cum_sum / vol_sum
    
    # Align to 1h
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_trend)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(vwap_4h_aligned[i]) or np.isnan(vwap_1d_aligned[i]):
            if position != 0:
                # Hold position outside session
                signals[i] = 0.20 if position == 1 else -0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price crosses below 1d VWAP or 4h trend turns bearish
            if close[i] < vwap_1d_aligned[i] or close[i] < vwap_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: price crosses above 1d VWAP or 4h trend turns bullish
            if close[i] > vwap_1d_aligned[i] or close[i] > vwap_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long: 4h bullish (price > 4h VWAP) + pullback to 1d VWAP then close above
            if (close[i] > vwap_4h_aligned[i] and  # 4h bullish trend
                close[i-1] <= vwap_1d_aligned[i-1] and close[i] > vwap_1d_aligned[i]):  # pullback to 1d VWAP
                position = 1
                signals[i] = 0.20
            # Short: 4h bearish (price < 4h VWAP) + pullback to 1d VWAP then close below
            elif (close[i] < vwap_4h_aligned[i] and  # 4h bearish trend
                  close[i-1] >= vwap_1d_aligned[i-1] and close[i] < vwap_1d_aligned[i]):  # pullback to 1d VWAP
                position = -1
                signals[i] = -0.20
    
    return signals