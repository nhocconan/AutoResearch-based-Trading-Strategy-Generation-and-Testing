#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike
# Uses 4h EMA50 for trend filter and 4h Camarilla levels (R1/S1) for breakout entries
# Entry: Long when price breaks above R1 AND price > 4h EMA50 (uptrend) AND volume spike
#        Short when price breaks below S1 AND price < 4h EMA50 (downtrend) AND volume spike
# Exit: Price crosses 4h EMA50 (trend reversal) OR price reverts to daily VWAP (mean reversion)
# Works in both bull and bear markets by trading breakouts with 4h trend filter
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Session filter: 08-20 UTC to reduce noise trades
# Discrete sizing 0.20 balances profit potential and fee drag

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precompute before loop)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (R1, S1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    pivot_4h = (high_4h + low_4h + close_4h_arr) / 3
    range_4h = high_4h - low_4h
    r1_4h = pivot_4h + range_4h * 1.1 / 12
    s1_4h = pivot_4h - range_4h * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily VWAP for mean reversion exit (more stable than Camarilla R2/S2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_1d = (typical_price_1d * df_1d['volume'].values).cumsum() / df_1d['volume'].values.cumsum()
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above R1 AND price > 4h EMA50 (uptrend) AND volume spike
            if (close[i] > r1_4h_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: Price breaks below S1 AND price < 4h EMA50 (downtrend) AND volume spike
            elif (close[i] < s1_4h_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price below 4h EMA50 (trend change) OR price reverts to daily VWAP (mean reversion)
            if close[i] < ema_50_4h_aligned[i] or close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price above 4h EMA50 (trend change) OR price reverts to daily VWAP (mean reversion)
            if close[i] > ema_50_4h_aligned[i] or close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals