#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_spike_v3
# Hypothesis: Camarilla pivot levels from 1d act as strong support/resistance. 
# Enter long at S1 with volume spike and bullish 1w trend; short at R1 with volume spike and bearish 1w trend.
# Uses 12h timeframe for low trade frequency (target: 50-150 total trades over 4 years).
# Works in bull/bear: 1w trend filter ensures alignment with major trend, pivots provide precise entry/exit.
# Volume spike confirms institutional interest at key levels.

name = "12h_camarilla_pivot_volume_spike_v3"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    # R4 = Close + ((High-Low) * 1.5000)
    # R3 = Close + ((High-Low) * 1.2500)
    # R2 = Close + ((High-Low) * 1.1666)
    # R1 = Close + ((High-Low) * 1.0833)
    # S1 = Close - ((High-Low) * 1.0833)
    # S2 = Close - ((High-Low) * 1.1666)
    # S3 = Close - ((High-Low) * 1.2500)
    # S4 = Close - ((High-Low) * 1.5000)
    
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.0833)
    s1 = prev_close - (rng * 1.0833)
    
    # Align pivots to 12h timeframe (wait for 1d close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1w trend filter (EMA 21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 2.0x 24-period average (24*12h = 12 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    if n >= vol_period:
        vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(21, vol_period) + 1
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below S1 or trend turns bearish
            if close[i] < s1_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above R1 or trend turns bullish
            if close[i] > r1_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at S1 with volume spike and bullish 1w trend
            if (abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.005 and  # Within 0.5% of S1
                volume_filter and 
                close[i] > ema_1w_aligned[i]):  # Bullish trend
                position = 1
                signals[i] = 0.25
            # Short entry: price at R1 with volume spike and bearish 1w trend
            elif (abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.005 and  # Within 0.5% of R1
                  volume_filter and 
                  close[i] < ema_1w_aligned[i]):  # Bearish trend
                position = -1
                signals[i] = -0.25
    
    return signals