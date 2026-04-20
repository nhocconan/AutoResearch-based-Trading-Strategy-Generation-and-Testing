#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and ADX trend filter.
# Donchian(20) breakouts capture strong momentum moves. Volume confirmation ensures institutional participation.
# ADX(14) > 25 filters for trending markets, avoiding range-bound periods where breakouts fail.
# This combination works in both bull and bear markets by following genuine trends with volume confirmation.
# Target: 25-40 trades per year to minimize fee drag while capturing significant moves.

name = "4h_Donchian20_1dVolume_ADXFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Volume confirmation ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma_1d > 0, vol_ma_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 4h Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h ADX(14) for trend strength ===
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - high[:-1]), np.abs(low[1:] - low[:-1])))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / np.where(atr_14 > 0, atr_14, np.nan)
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / np.where(atr_14 > 0, atr_14, np.nan)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) > 0, (plus_di_14 + minus_di_14), np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Get values
        high_val = prices['high'].iloc[i]
        low_val = prices['low'].iloc[i]
        close_val = prices['close'].iloc[i]
        donch_high = high_20[i]
        donch_low = low_20[i]
        adx_val = adx[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(donch_high) or np.isnan(donch_low) or np.isnan(adx_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume and trend confirmation
            if high_val > donch_high and vol_ratio_val > 1.3 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume and trend confirmation
            elif low_val < donch_low and vol_ratio_val > 1.3 and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Donchian low or trend weakens
            if close_val < donch_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Donchian high or trend weakens
            if close_val > donch_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals