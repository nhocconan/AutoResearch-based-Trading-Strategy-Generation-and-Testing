#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and ATR filter
# Primary timeframe 4h with HTF 1d Donchian channels (20-period) for major trend direction
# Volume confirmation (current 4h volume > 1.5x 20-period average) filters low-conviction breakouts
# ATR filter ensures sufficient volatility (ATR > 20-period average) to avoid choppy markets
# Fixed position size 0.25 to balance return and drawdown
# Target: 25-60 trades/year on 4h timeframe (100-240 total over 4 years)

name = "4h_1d_donchian_volume_atr_v2"
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
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF data to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR filter (20-period average for 4h ATR)
    # Calculate 4h ATR first
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_4h = pd.Series(atr_14_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_ma_20_4h[i]) or atr_14_1d_aligned[i] <= 0 or atr_ma_20_4h[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: 4h ATR > 20-period average ATR
        vol_filter = atr_14_4h[i] > atr_ma_20_4h[i]
        
        if not (volume_confirmed and vol_filter):
            signals[i] = 0.0
            continue
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to midpoint of Donchian channel
            midpoint = (high_20_aligned[i] + low_20_aligned[i]) / 2.0
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to midpoint of Donchian channel
            midpoint = (high_20_aligned[i] + low_20_aligned[i]) / 2.0
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with volume and volatility confirmation
            if close[i] > high_20_aligned[i]:
                position = 1
                signals[i] = position_size
            elif close[i] < low_20_aligned[i]:
                position = -1
                signals[i] = -position_size
    
    return signals