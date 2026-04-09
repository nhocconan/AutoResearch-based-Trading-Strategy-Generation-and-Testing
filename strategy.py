#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d ATR volatility filter + volume confirmation
# Donchian(20) breakout provides clear structure; 1d ATR filter avoids low volatility chop
# Volume confirmation (>1.5x 20-period average) filters false breakouts
# Works in bull/bear: breakouts capture momentum, ATR filter avoids whipsaws in range
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # ATR filter: only trade when 1d ATR > 0.5 * 20-period ATR average (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
        atr_filter = atr_1d_aligned[i] > 0.5 * atr_ma_20[i] if not np.isnan(atr_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit on Donchian lower band retracement
            if close[i] < low_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Donchian upper band retracement
            if close[i] > high_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume and ATR confirmation
            if volume_confirmed and atr_filter:
                if close[i] > high_ma_20[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < low_ma_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals