#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # EMA200 on 4h for long-term trend
    close_4h_series = pd.Series(close_4h)
    ema200_4h = close_4h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 1d data for pivot points ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points: P, R1, S1
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1_1d = pivot_1d + range_hl
    s1_1d = pivot_1d - range_hl
    
    # Align daily pivot levels to 1h timeframe
    pivot_1h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(pivot_1h[i]) or np.isnan(r1_1h[i]) or
            np.isnan(s1_1h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema200 = ema200_4h_aligned[i]
        pivot_level = pivot_1h[i]
        r1_level = r1_1h[i]
        s1_level = s1_1h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to pivot level (mean reversion)
            if price <= pivot_level:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to pivot level (mean reversion)
            if price >= pivot_level:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above EMA200 (uptrend) and breaks above R1 with volume spike
            if price > ema200 and price > r1_level and vol_spike:
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price below EMA200 (downtrend) and breaks below S1 with volume spike
            elif price < ema200 and price < s1_level and vol_spike:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Pivot_R1_S1_Breakout_Volume_EMA200"
timeframe = "1h"
leverage = 1.0