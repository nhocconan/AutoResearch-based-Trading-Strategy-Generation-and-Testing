#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation and volatility filter
# - Long when price breaks above previous weekly high with volume > 1.5x 20-day average and rising ATR
# - Short when price breaks below previous weekly low with volume > 1.5x 20-day average and rising ATR
# - Uses rising ATR as volatility filter to avoid whipsaws in low volatility periods
# - Exits on opposite breakout (mean reversion tendency in ranging markets)
# - Position size 0.25 to balance risk and returns
# - Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag
# - Works in both bull and bear markets by capturing breakouts with volatility confirmation

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
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get previous weekly high/low for breakout levels
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        
        # Create arrays for alignment (constant values for the weekly period)
        high_array = np.full(len(df_1w), prev_high)
        low_array = np.full(len(df_1w), prev_low)
        
        # Align to daily timeframe
        high_1d = align_htf_to_ltf(prices, df_1w, high_array)[i]
        low_1d = align_htf_to_ltf(prices, df_1w, low_array)[i]
        
        if position == 0:
            # Long: Break above previous weekly high with volume and volatility filter
            if (close[i] > high_1d and 
                volume[i] > vol_ma[i] * 1.5 and
                i >= 28 and atr[i] > atr[i-14]):
                position = 1
                signals[i] = position_size
            # Short: Break below previous weekly low with volume and volatility filter
            elif (close[i] < low_1d and 
                  volume[i] > vol_ma[i] * 1.5 and
                  i >= 28 and atr[i] > atr[i-14]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous weekly low (mean reversion in ranging markets)
            if close[i] < low_1d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous weekly high (mean reversion in ranging markets)
            if close[i] > high_1d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_DonchianBreakout_Volume_VolatilityFilter"
timeframe = "1d"
leverage = 1.0