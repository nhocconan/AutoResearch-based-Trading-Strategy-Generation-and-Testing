#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and
#   chop regime filter. Long when price breaks above 20-period high with volume
#   spike and chop > 61.8 (range); short when breaks below 20-period low with
#   volume spike and chop > 61.8. This structure captures volatility expansion
#   in ranging markets, works in both bull and bear regimes by filtering for
#   high-probability mean-reversion bursts. Target: 20-50 trades/year.
name = "12h_Donchian20_VolumeChop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Chop index (14-period) for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    chop = 100 * np.log10(atr14.sum() / (highest_high20 - lowest_low20)) / np.log10(14)
    # Fix: correct chop calculation using rolling sum
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high20 - lowest_low20
    chop = np.where(hh_ll_diff != 0, 100 * np.log10(atr_sum / hh_ll_diff) / np.log10(14), 50)
    
    # 1-day volume average for spike detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_avg20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_avg20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 12h volume > 2x daily average volume (approximation)
        vol_spike = volume[i] > (2.0 * vol_avg20_1d_aligned[i])
        
        if position == 0:
            # Long: breakout above 20-period high, volume spike, chop > 61.8 (range)
            long_cond = (close[i] > high_max20[i]) and vol_spike and (chop[i] > 61.8)
            # Short: breakdown below 20-period low, volume spike, chop > 61.8
            short_cond = (close[i] < low_min20[i]) and vol_spike and (chop[i] > 61.8)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to 20-period low or chop drops (trend emerging)
            if close[i] < low_min20[i] or chop[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to 20-period high or chop drops
            if close[i] > high_max20[i] or chop[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals