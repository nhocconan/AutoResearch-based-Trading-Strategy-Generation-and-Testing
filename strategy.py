#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour Donchian channel breakout with volume confirmation and volatility filter
# - Long when price breaks above previous 12h high with volume > 1.6x 48-period average and rising volatility
# - Short when price breaks below previous 12h low with volume > 1.6x 48-period average and rising volatility
# - Uses rising ATR as volatility filter to avoid whipsaws in low volatility
# - Exits on opposite breakout
# - Position size 0.28 to balance risk and returns
# - Target: 80-180 trades over 4 years (20-45/year) to avoid excessive fees

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate ATR for volatility filter
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
    position_size = 0.28
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get previous 12h high/low for breakout levels
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        
        # Create arrays for alignment (constant values for the 12h period)
        high_array = np.full(len(df_12h), prev_high)
        low_array = np.full(len(df_12h), prev_low)
        
        # Align to 4h timeframe
        high_4h = align_htf_to_ltf(prices, df_12h, high_array)[i]
        low_4h = align_htf_to_ltf(prices, df_12h, low_array)[i]
        
        if position == 0:
            # Long: Break above previous 12h high with volume and volatility filter
            if (close[i] > high_4h and 
                volume[i] > vol_ma[i] * 1.6 and
                i >= 28 and atr[i] > atr[i-14]):
                position = 1
                signals[i] = position_size
            # Short: Break below previous 12h low with volume and volatility filter
            elif (close[i] < low_4h and 
                  volume[i] > vol_ma[i] * 1.6 and
                  i >= 28 and atr[i] > atr[i-14]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous 12h low
            if close[i] < low_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous 12h high
            if close[i] > high_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_DonchianBreakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0