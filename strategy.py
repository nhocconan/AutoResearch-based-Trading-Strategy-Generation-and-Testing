#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week Donchian channel breakout with volume confirmation and volatility filter
# - Long when price breaks above previous week's high with volume > 1.8x 48-period average
# - Short when price breaks below previous week's low with volume > 1.8x 48-period average
# - Requires volatility expansion: ATR(14) > 1.3x ATR(14) 14 periods ago
# - Exits on opposite breakout
# - Position size 0.25 to manage risk and reduce churn
# - Target: 50-120 trades over 4 years (12-30/year) to avoid fee drag
# - Using weekly timeframe for better trend alignment and reduced noise

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate ATR for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 48-period average (2 days of 4h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get previous week's high/low for breakout levels
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        
        # Create arrays for alignment (constant values for the week)
        high_array = np.full(len(df_1w), prev_high)
        low_array = np.full(len(df_1w), prev_low)
        
        # Align to 4h timeframe
        high_4h = align_htf_to_ltf(prices, df_1w, high_array)[i]
        low_4h = align_htf_to_ltf(prices, df_1w, low_array)[i]
        
        if position == 0:
            # Long: Break above previous week's high with volume and volatility expansion
            if (close[i] > high_4h and 
                volume[i] > vol_ma[i] * 1.8 and
                i >= 28 and atr[i] > atr[i-14] * 1.3):
                position = 1
                signals[i] = position_size
            # Short: Break below previous week's low with volume and volatility expansion
            elif (close[i] < low_4h and 
                  volume[i] > vol_ma[i] * 1.8 and
                  i >= 28 and atr[i] > atr[i-14] * 1.3):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous week's low
            if close[i] < low_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous week's high
            if close[i] > high_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1w_DonchianBreakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0