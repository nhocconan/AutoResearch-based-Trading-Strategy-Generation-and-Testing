#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR-based volume spike + chop regime filter
# Uses 1d ATR to normalize volume spike detection (more robust than fixed multiplier)
# Chop regime: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (fade breakout)
# Discrete sizing 0.25 to limit fee drift. Target: 75-200 trades over 4 years.

name = "4h_1d_donchian_atr_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # --- 1d Indicators ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) using Wilder's smoothing
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Volume spike: current 1d volume > 2.0 x 20-day average volume
    volume_spike_1d = volume_1d > (2.0 * avg_volume_1d)
    
    # 1d Choppiness Index (CHOP) over 14 periods
    atr_14_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0,
                       100 * np.log10(atr_14_sum / range_14) / np.log10(14),
                       50)
    
    # Align 1d indicators to 4h (wait for 1d bar close)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # --- 4h Donchian Channels (20-period) ---
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR regime shifts to ranging
            if close[i] < lowest_low[i] or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR regime shifts to ranging
            if close[i] > highest_high[i] or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: 1d volume spike
            vol_confirmed = volume_spike_1d_aligned[i]
            
            if chop_1d_aligned[i] < 38.2:  # Trending regime: follow breakout
                if close[i] > highest_high[i] and vol_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low[i] and vol_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif chop_1d_aligned[i] > 61.8:  # Ranging regime: fade breakout
                if close[i] < lowest_low[i] and vol_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > highest_high[i] and vol_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals