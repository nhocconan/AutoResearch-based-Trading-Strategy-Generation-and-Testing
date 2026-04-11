#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter with Donchian(20) breakout and volume confirmation
# In choppy markets (CHOP > 61.8): mean reversion at Donchian bands (sell upper band, buy lower band)
# In trending markets (CHOP < 38.2): breakout continuation (buy upper band break, sell lower band break)
# Uses 1h timeframe for entries, 1d for regime and Donchian, 1h for volume confirmation
# Designed to work in both bull and bear markets by adapting to regime
# Target: 20-50 trades/year on 1h timeframe with regime adaptation to reduce whipsaw

name = "1d_1h_chop_regime_donchian_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for regime and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.nanmean(tr[1:15])  # Skip first NaN
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr = np.zeros_like(atr)
    for i in range(len(sum_atr)):
        if i < 14:
            sum_atr[i] = np.nan
        elif i == 14:
            sum_atr[i] = np.nansum(atr[1:15])
        else:
            sum_atr[i] = sum_atr[i-1] - atr[i-14] + atr[i]
    
    # Choppiness Index: 100 * log10(sum_atr / (max_high - min_low)) / log10(14)
    max_high = np.zeros_like(high_1d)
    min_low = np.zeros_like(low_1d)
    for i in range(len(max_high)):
        if i < 14:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.nanmax(high_1d[i-13:i+1])
            min_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d indicators to 1h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Calculate 1h volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Regime filters
        is_choppy = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        # Price levels
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        
        # Entry conditions based on regime
        if is_choppy:
            # Mean reversion: sell at upper band, buy at lower band
            long_entry = (close[i] < lower_band) and volume_filter
            short_entry = (close[i] > upper_band) and volume_filter
            # Exit when price returns to middle
            long_exit = (close[i] > donchian_mid_aligned[i])
            short_exit = (close[i] < donchian_mid_aligned[i])
        else:  # trending or neutral
            # Breakout continuation: buy upper band break, sell lower band break
            long_entry = (close[i] > upper_band) and volume_filter
            short_entry = (close[i] < lower_band) and volume_filter
            # Exit when price returns to middle or opposite band touch
            long_exit = (close[i] < donchian_mid_aligned[i]) or (close[i] < lower_band)
            short_exit = (close[i] > donchian_mid_aligned[i]) or (close[i] > upper_band)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals