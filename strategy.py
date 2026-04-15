#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily ATR ratio (current vs 50-period average) for regime filter
    atr_ma_50 = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14_1d / (atr_ma_50 + 1e-10)
    
    # Align HTF ATR ratio to 12h timeframe
    atr_ratio_12h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 12h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_12h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in low volatility environments (ATR ratio < 0.8)
        # This avoids whipsaws in high volatility periods
        if atr_ratio_12h[i] >= 0.8:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: price breaks above Donchian upper band with volume confirmation
        # Short: price breaks below Donchian lower band with volume confirmation
        # Position size: 0.25 (discrete to minimize fee churn)
        
        if (close[i] > highest_20[i] and  # Breakout above upper Donchian
            volume_ratio[i] > 1.5):       # Volume confirmation
            signals[i] = 0.25
            
        elif (close[i] < lowest_20[i] and  # Breakdown below lower Donchian
              volume_ratio[i] > 1.5):      # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_LowVol_Filter"
timeframe = "12h"
leverage = 1.0