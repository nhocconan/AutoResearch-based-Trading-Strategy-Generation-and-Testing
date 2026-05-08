#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d chop regime
# Long when price breaks above 20-period high AND volume > 1.5x 10-period avg AND chop > 61.8 (range)
# Short when price breaks below 20-period low AND volume > 1.5x 10-period avg AND chop > 61.8 (range)
# Exit when price returns to 10-period moving average
# Uses 1d data for volume and chop to avoid noise, 12h for breakout timing
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Donchian20_1dVolume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average (10-period)
    volume_1d = df_1d['volume'].values
    vol_avg_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_10_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10)
    
    # Calculate 1d chopiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr3 = np.abs(np.roll(close_1d, 1) - close_1d)
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    atr_sum_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        start = max(0, i-14)
        atr_sum_1d[i] = np.sum(atr_1d[start:i+1]) if i >= start else 0
    
    denominator = max_high_1d - min_low_1d
    chop_1d = np.where(denominator != 0, 100 * np.log10(atr_sum_1d / denominator) / np.log10(14), 50)
    chop_1d = np.where(denominator == 0, 50, chop_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_avg_10_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume_1d[i] / vol_avg_10_aligned[i] if vol_avg_10_aligned[i] > 0 else 0
        
        if position == 0:
            # Enter long: break above Donchian high + volume spike + chop > 61.8
            if close[i] > donch_high[i] and vol_ratio > 1.5 and chop_1d_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low + volume spike + chop > 61.8
            elif close[i] < donch_low[i] and vol_ratio > 1.5 and chop_1d_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to 10-period MA
            if close[i] <= ma_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to 10-period MA
            if close[i] >= ma_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals