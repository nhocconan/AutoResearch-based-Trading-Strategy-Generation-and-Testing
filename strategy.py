#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout (20-period) with volume confirmation and ADX trend filter.
# Long when price breaks above weekly Donchian high with volume spike and ADX > 25.
# Short when price breaks below weekly Donchian low with volume spike and ADX > 25.
# Weekly trend filter ensures we trade with the higher timeframe momentum, reducing false signals in choppy markets.
# Position size: 0.25. Target: 15-25 trades/year per symbol.
name = "1d_WeeklyDonchian20_Volume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    donch_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe (only use after weekly bar closes)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Weekly ADX for trend strength (14-period)
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate True Range components
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift(1))).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(df_1w['high']).diff()
    dm_minus = -pd.Series(df_1w['low']).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values (14-period)
    tr_ma = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high, ADX > 25, volume spike
            if (close[i] > donch_high_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low, ADX > 25, volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low or ADX < 20
            if (close[i] < donch_low_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high or ADX < 20
            if (close[i] > donch_high_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals