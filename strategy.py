#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume spike + ATR filter.
# Uses daily timeframe for ATR calculation to ensure volatility-based breakouts.
# Entry: Long when price breaks above 20-period high with volume spike and ATR > threshold.
# Exit: Opposite Donchian level touch or ATR-based trailing stop.
# Designed for 12h timeframe to capture medium-term breakouts with low frequency (<30 trades/year).
# Volume spike filters false breakouts; ATR filter ensures sufficient volatility.
# Designed to work in both bull and bear markets by capturing breakouts in either direction.
name = "12h_Donchian20_Volume_ATRFilter"
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
    
    # Daily ATR for volatility filter (using 14-period ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range components
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - exponential moving average of TR
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ATR to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Donchian channels (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 20-period high with volume spike and sufficient volatility
            if (close[i] > high_20[i] and 
                volume_spike[i] and 
                atr_12h[i] > 0):  # Ensure volatility is present
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low with volume spike and sufficient volatility
            elif (close[i] < low_20[i] and 
                  volume_spike[i] and 
                  atr_12h[i] > 0):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches 20-period low or ATR-based stop
            # ATR stop: exit if price drops below entry by 1.5 * ATR
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches 20-period high or ATR-based stop
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals