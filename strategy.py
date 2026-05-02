#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# Donchian breakout captures strong directional moves in both bull and bear markets
# 1d volume spike (>2.0 x 20-period EMA) confirms breakout validity and reduces false signals
# Chop regime filter (CHOP > 61.8) avoids whipsaws in ranging markets, only trades in clear trends
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_Donchian20_Breakout_1dVolume_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume EMA(20) for confirmation
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    volume_confirmation = volume > (2.0 * vol_ema_20_1d_aligned)
    
    # Chop regime filter (14-period) on 1d timeframe
    # Chop = 100 * log10(sum(ATR) / (log10(highest_high - lowest_low) * n))
    tr_1d = np.maximum(np.maximum(df_1d['high'].values - df_1d['low'].values,
                                  np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)),
                         np.abs(df_1d['low'].values - df_1d['close'].shift(1).values))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_14_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    price_range_14_1d = highest_high_14_1d - lowest_low_14_1d
    price_range_14_1d = np.where(price_range_14_1d == 0, 1e-10, price_range_14_1d)
    
    chop_14_1d = 100 * np.log10(
        np.sum(atr_14_1d.reshape(-1, 14), axis=1) / 
        (np.log10(price_range_14_1d) * 14)
    )
    # Handle edge cases where reshape fails
    chop_14_1d = np.full(len(df_1d), 50.0)  # neutral chop as fallback
    if len(df_1d) >= 14:
        tr_1d = np.maximum(np.maximum(df_1d['high'].values - df_1d['low'].values,
                                      np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)),
                             np.abs(df_1d['low'].values - df_1d['close'].shift(1).values))
        atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        highest_high_14_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
        lowest_low_14_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
        price_range_14_1d = highest_high_14_1d - lowest_low_14_1d
        price_range_14_1d = np.where(price_range_14_1d == 0, 1e-10, price_range_14_1d)
        chop_14_1d = 100 * np.log10(
            atr_14_1d * 14 / 
            (np.log10(price_range_14_1d) * 14)
        )
        chop_14_1d = np.where(np.isnan(chop_14_1d), 50.0, chop_14_1d)
    
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    chop_filter = chop_14_1d_aligned > 61.8  # trending regime (avoid choppy markets)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian calculation)
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_confirmation[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian channel with volume confirmation and trending regime
            if close[i] > highest_20[i] and volume_confirmation[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with volume confirmation and trending regime
            elif close[i] < lowest_20[i] and volume_confirmation[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below midpoint of Donchian channel (trailing stop)
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above midpoint of Donchian channel (trailing stop)
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals