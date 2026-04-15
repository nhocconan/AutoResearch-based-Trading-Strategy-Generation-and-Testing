#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 1d Donchian(20) upper band + volume > 1.5x 20-period avg + 1d ADX > 25
# Short when price breaks below 1d Donchian(20) lower band + volume > 1.5x 20-period avg + 1d ADX > 25
# Uses 1d price structure (Donchian channels) and 1d ADX for trend strength on 12h chart
# Designed for low trade frequency (12-25/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and trend strength filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) and ADX (14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian Channel (20-period)
    donchian_upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_sum = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).sum().values
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_sum / tr_sum)
    di_minus = 100 * (dm_minus_sum / tr_sum)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align all 1d indicators to 12h timeframe
    donchian_upper_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20)
    donchian_lower_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 30
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_20_aligned[i]) or np.isnan(donchian_lower_20_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Donchian upper band
        # 2. Volume confirmation
        # 3. 1d ADX > 25 (strong trend)
        if (close[i] > donchian_upper_20_aligned[i]) and vol_confirm and (adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Donchian lower band
        # 2. Volume confirmation
        # 3. 1d ADX > 25 (strong trend)
        elif (close[i] < donchian_lower_20_aligned[i]) and vol_confirm and (adx_aligned[i] > 25):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_Volume_1dADX_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0