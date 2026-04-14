#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Donchian channel breakout with volume confirmation and ATR filter
# - Long when price breaks above previous day's high with volume > 1.5x 24-period average
# - Short when price breaks below previous day's low with volume > 1.5x 24-period average
# - Requires ATR(20) > 1.2x ATR(20) 20 periods ago to ensure volatility expansion
# - Exits on opposite breakout or ATR contraction (< 0.8x ATR 20 periods ago)
# - Position size 0.25 to manage risk and reduce churn
# - Target: 50-150 trades over 4 years (12-38/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ATR for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: 24-period average (1 day of 4h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get previous day's high/low for breakout levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        
        # Create arrays for alignment (constant values for the day)
        high_array = np.full(len(df_1d), prev_high)
        low_array = np.full(len(df_1d), prev_low)
        
        # Align to 4h timeframe
        high_4h = align_htf_to_ltf(prices, df_1d, high_array)[i]
        low_4h = align_htf_to_ltf(prices, df_1d, low_array)[i]
        
        if position == 0:
            # Long: Break above previous day's high with volume and volatility expansion
            if (close[i] > high_4h and 
                volume[i] > vol_ma[i] * 1.5 and
                i >= 40 and atr[i] > atr[i-20] * 1.2):
                position = 1
                signals[i] = position_size
            # Short: Break below previous day's low with volume and volatility expansion
            elif (close[i] < low_4h and 
                  volume[i] > vol_ma[i] * 1.5 and
                  i >= 40 and atr[i] > atr[i-20] * 1.2):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous day's low or ATR contraction
            if close[i] < low_4h or (i >= 40 and atr[i] < atr[i-20] * 0.8):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous day's high or ATR contraction
            if close[i] > high_4h or (i >= 40 and atr[i] < atr[i-20] * 0.8):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_DonchianBreakout_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0