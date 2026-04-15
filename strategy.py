#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Volatility Filter and Volume Confirmation
# Long when price breaks above Donchian(20) high with volatility expansion and volume spike
# Short when price breaks below Donchian(20) low with volatility expansion and volume spike
# Volatility filter: ATR(10) > 1.5x ATR(30) to ensure breakouts occur during high volatility
# Volume confirmation: current volume > 2.0x median of last 20 bars
# This structure reduces false breakouts and focuses on high-probability moves
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean()
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean()
    
    # 1-day volatility filter using ATR ratio
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1d = high_1d - low_1d
    tr2d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2d[0] = tr1d[0]
    tr3d[0] = tr1d[0]
    tr_d = np.maximum(tr1d, np.maximum(tr2d, tr3d))
    atr_10_1d = pd.Series(tr_d).rolling(window=10, min_periods=10).mean()
    atr_30_1d = pd.Series(tr_d).rolling(window=30, min_periods=30).mean()
    atr_ratio_1d = atr_10_1d / (atr_30_1d + 1e-10)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d.values)
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: break above Donchian high, volatility expansion, volume spike
        if (close[i] > high_20[i] and 
            atr_ratio_1d_aligned[i] > 1.5 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: break below Donchian low, volatility expansion, volume spike
        elif (close[i] < low_20[i] and 
              atr_ratio_1d_aligned[i] > 1.5 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel or volatility contracts
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < (high_20[i] + low_20[i]) / 2 or atr_ratio_1d_aligned[i] <= 1.2)) or
               (signals[i-1] == -0.25 and (close[i] > (high_20[i] + low_20[i]) / 2 or atr_ratio_1d_aligned[i] <= 1.2)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_VolATR_1dFilter"
timeframe = "4h"
leverage = 1.0