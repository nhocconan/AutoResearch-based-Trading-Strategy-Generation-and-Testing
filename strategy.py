#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1-day ATR filter and volume confirmation.
# Donchian channels capture breakouts; ATR filter avoids high volatility false signals;
# volume confirms institutional participation. Designed for 4h to balance signal quality
# and trade frequency (~20-40 trades/year). Works in bull/bear via volatility-adjusted breaks.
name = "4h_Donchian20_ATR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day ATR (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 4h timeframe (waits for prior day close)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Avoid trading in extremely low volatility (ATR near zero)
        if atr_aligned[i] < 0.0001 * close[i]:  # less than 0.01% of price
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with volatility filter and volume
            if (close[i] > high_roll[i] and 
                atr_aligned[i] < 0.02 * close[i] and  # ATR < 2% of price
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volatility filter and volume
            elif (close[i] < low_roll[i] and 
                  atr_aligned[i] < 0.02 * close[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to midpoint or volatility spikes
            midpoint = (high_roll[i] + low_roll[i]) / 2
            if close[i] < midpoint or atr_aligned[i] > 0.03 * close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to midpoint or volatility spikes
            midpoint = (high_roll[i] + low_roll[i]) / 2
            if close[i] > midpoint or atr_aligned[i] > 0.03 * close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals