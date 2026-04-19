# 1h_Pivot_R1S1_Breakout_Volume_Spike_v1
# Hypothesis: 1h breakout with volume spike at 4h/1d pivot levels captures breakouts with confirmation, suitable for 1h timeframe with proper filtering to avoid overtrading.
# Uses 4h trend filter and session filter to reduce noise, targeting 15-30 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Pivot_R1S1_Breakout_Volume_Spike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h pivot levels from previous 4h bar
    prev_close_4h = np.roll(close_4h, 1)
    prev_close_4h[0] = np.nan
    prev_high_4h = np.roll(high_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h = np.roll(low_4h, 1)
    prev_low_4h[0] = np.nan
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    r1_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12.0
    s1_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12.0
    
    # Calculate 1d pivot levels from previous day
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d = np.roll(high_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d = np.roll(low_1d, 1)
    prev_low_1d[0] = np.nan
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    s1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    
    # Align to 1h timeframe
    pivot_4h_1h = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_1h = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_1h = align_htf_to_ltf(prices, df_4h, s1_4h)
    pivot_1d_1h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h trend filter: close > EMA20 for long, close < EMA20 for short
    close_4h_series = pd.Series(close_4h)
    ema20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_1h = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h_1h[i]) or np.isnan(r1_4h_1h[i]) or np.isnan(s1_4h_1h[i]) or \
           np.isnan(pivot_1d_1h[i]) or np.isnan(r1_1d_1h[i]) or np.isnan(s1_1d_1h[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(ema20_4h_1h[i]):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0 and in_session:
            # Long: Price breaks above 4h R1 with volume spike, above 1d pivot, and above 4h EMA20
            if price > r1_4h_1h[i] and volume_spike and price > pivot_1d_1h[i] and price > ema20_4h_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h S1 with volume spike, below 1d pivot, and below 4h EMA20
            elif price < s1_4h_1h[i] and volume_spike and price < pivot_1d_1h[i] and price < ema20_4h_1h[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 4h S1 (reversal signal) or outside session
            if price < s1_4h_1h[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price returns above 4h R1 (reversal signal) or outside session
            if price > r1_4h_1h[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals