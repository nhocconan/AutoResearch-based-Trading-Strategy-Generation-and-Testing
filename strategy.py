#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R1, close > 12h EMA50, volume > 1.8x 20-bar average
# Short when price breaks below Camarilla S1, close < 12h EMA50, volume > 1.8x 20-bar average
# Uses Camarilla pivot levels for structure, 12h EMA50 for trend filter, volume for momentum confirmation
# Designed for low trade frequency (~19-50/year on 4h) to minimize fee drag
# Works in bull (breakouts with rising volume in uptrend) and bear (breakdowns with rising volume in downtrend)
# Adding 12h trend filter (vs previous 1d) to reduce whipsaw and improve signal quality

name = "4h_Camarilla_R1S1_Volume_12hEMA50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate previous 12h bar's Camarilla levels (R1, S1)
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)  # Camarilla R1
    s1 = pivot - (range_hl * 1.1 / 12.0)  # Camarilla S1
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume confirmation (1.8x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20) + 1  # EMA50(12h) + Camarilla + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R1, close > 12h EMA50, volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Camarilla S1, close < 12h EMA50, volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S1 or close < 12h EMA50 (trend failure)
            if (close[i] < s1_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R1 or close > 12h EMA50 (trend failure)
            if (close[i] > r1_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals