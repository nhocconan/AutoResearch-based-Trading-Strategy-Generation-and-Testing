# ITERATION 44976: 12h Donchian Breakout + Volume + Volatility Regime
# Hypothesis: Breakouts of 20-period Donchian channel on 12h timeframe, filtered by volume surge and low volatility regime (choppy markets avoided), work in both bull and bear markets by capturing strong directional moves. Uses 1d timeframe for volatility regime filter to avoid false breakouts in chop. Target: 50-150 total trades over 4 years (12-37/year).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h price data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for volatility regime filter (ATR-based chop filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12-period ATR on 1d for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12_1d = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    
    # Calculate 50-period SMA of ATR for regime threshold (high ATR = trending, low ATR = chop)
    atr_ma_50_1d = pd.Series(atr_12_1d).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: 1 = trending (high volatility), 0 = chop (low volatility)
    vol_regime = (atr_12_1d > atr_ma_50_1d).astype(float)
    
    # Calculate Donchian channels (20-period) on 12h
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Al indicators to 12h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    
    # Volume average (20-period) on 12h for confirmation
    vol_avg_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # volume is already 12h aligned via prices
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size: 25% of capital
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_regime_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(vol_avg_12h[i])):
            continue
        
        # Long entry: price breaks above upper Donchian + volume surge + trending regime
        if (close[i] > highest_20_aligned[i] and 
            volume[i] > 2.0 * vol_avg_12h[i] and 
            vol_regime_aligned[i] > 0.5 and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower Donchian + volume surge + trending regime
        elif (close[i] < lowest_20_aligned[i] and 
              volume[i] > 2.0 * vol_avg_12h[i] and 
              vol_regime_aligned[i] > 0.5 and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or loss of volatility regime (avoid whipsaw in chop)
        elif position == 1 and vol_regime_aligned[i] <= 0.5:
            position = 0
            signals[i] = 0.0
        elif position == -1 and vol_regime_aligned[i] <= 0.5:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_VolRegime"
timeframe = "12h"
leverage = 1.0