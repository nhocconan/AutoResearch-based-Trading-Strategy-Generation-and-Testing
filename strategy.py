#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Choppiness_Index_Donchian_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels (1-period breakout)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian(20) channels: upper/lower bands
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Choppiness Index (14-period) from 1d data
    # CHOP = 100 * log10(sum(TR over n) / (HHV - LLV)) / log10(n)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hhvl = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llvl = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hhvl - llvl)) / np.log10(14)
    
    # Align Choppiness Index to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper, chop > 61.8 (ranging), price above EMA50, volume spike
            long_cond = (close[i] > donchian_upper_aligned[i] and
                        chop_aligned[i] > 61.8 and
                        close[i] > ema50_1d_aligned[i] and
                        volume_spike[i])
            
            # Short: Price breaks below Donchian lower, chop > 61.8 (ranging), price below EMA50, volume spike
            short_cond = (close[i] < donchian_lower_aligned[i] and
                         chop_aligned[i] > 61.8 and
                         close[i] < ema50_1d_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian lower OR chop < 38.2 (trending) OR price crosses below EMA50
            if (close[i] < donchian_lower_aligned[i] or
                chop_aligned[i] < 38.2 or
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian upper OR chop < 38.2 (trending) OR price crosses above EMA50
            if (close[i] > donchian_upper_aligned[i] or
                chop_aligned[i] < 38.2 or
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals