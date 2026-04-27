#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev * 2) / 4
    range_ = high_prev - low_prev
    
    # Resistance and Support levels (R1 and S1 only)
    r1 = pivot + range_ * 1.083
    s1 = pivot - range_ * 1.083
    
    # Align pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert)
    atr = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0))
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)
    chop_range = chop > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, pivots, volume MA, chop
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_range[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        in_range = chop_range[i]
        
        if position == 0:
            # Long: price touches S1 + weekly uptrend + volume spike + ranging market
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and 
                close[i] > ema_trend and vol_spike_val and in_range):
                signals[i] = size
                position = 1
            # Short: price touches R1 + weekly downtrend + volume spike + ranging market
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and 
                  close[i] < ema_trend and vol_spike_val and in_range):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R1 or trend reverses or exits ranging market
            if high[i] >= r1_aligned[i] or close[i] < ema_trend or not in_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches S1 or trend reverses or exits ranging market
            if low[i] <= s1_aligned[i] or close[i] > ema_trend or not in_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_S1R1_1wEMA34_Trend_VolumeSpike_Range_v1"
timeframe = "1d"
leverage = 1.0