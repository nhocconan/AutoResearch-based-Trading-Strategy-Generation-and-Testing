#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter.
# Uses Donchian breakouts for structure, volume spike for conviction, and chop filter to avoid whipsaws in ranging markets.
# Works in bull (buy breakouts with volume) and bear (sell breakdowns with volume).
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Discrete position sizing (0.25) to minimize fee churn.

name = "4h_Donchian20_Breakout_Volume_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(np.maximum(df_1d['high'] - df_1d['low'], 
                               np.abs(df_1d['high'] - df_1d['close'].shift(1))),
                      np.abs(df_1d['low'] - df_1d['close'].shift(1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian channels (20-period) from previous candles
    # Shift by 1 to avoid look-ahead: use previous 20 candles for current breakout
    donchian_high = pd.Series(high).shift(1).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).shift(1).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Donchian, volume median, and chop
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.8)
        
        # Chop regime filter: only trade when CHOP < 50 (less choppy = more trending)
        # Avoid trading in high chop (ranging) markets
        chop_filter = chop_aligned[i] < 50
        
        # Donchian breakout conditions
        breakout_up = curr_close > donchian_high[i]   # break above upper Donchian
        breakout_down = curr_close < donchian_low[i]  # break below lower Donchian
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND volume confirmation AND chop filter
            if breakout_up and volume_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down AND volume confirmation AND chop filter
            elif breakout_down and volume_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals