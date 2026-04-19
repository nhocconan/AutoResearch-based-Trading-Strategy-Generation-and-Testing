#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover (8/21) with 4h EMA(34) trend filter and volume spike confirmation.
# EMA(8) crossing above EMA(21) indicates short-term bullish momentum; below indicates bearish.
# 4h EMA(34) filters trades to align with higher timeframe trend, reducing counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms breakout strength.
# Designed for 1h timeframe to capture medium-term moves with controlled frequency.
# Entry: Long when EMA8 > EMA21, EMA8 > 4h EMA34, and volume spike; Short when EMA8 < EMA21, EMA8 < 4h EMA34, and volume spike.
# Exit: Opposite EMA crossover or loss of volume confirmation.
# Uses strict conditions to limit trades (~15-35/year) and avoid overtrading.
name = "1h_EMA8_21_4hEMA34_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(8) and EMA(21) on 1h
    close_s = pd.Series(close)
    ema8 = close_s.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h EMA(34) - get once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: EMA8 > EMA21, above 4h EMA34 trend, with volume spike
            if (ema8[i] > ema21[i] and 
                ema8[i] > ema34_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: EMA8 < EMA21, below 4h EMA34 trend, with volume spike
            elif (ema8[i] < ema21[i] and 
                  ema8[i] < ema34_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if EMA8 < EMA21 or loses volume confirmation
            if (ema8[i] < ema21[i]) or (not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if EMA8 > EMA21 or loses volume confirmation
            if (ema8[i] > ema21[i]) or (not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals