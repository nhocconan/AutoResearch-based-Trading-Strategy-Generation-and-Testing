#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Donchian channel breakout with volume confirmation and ATR-based volatility filter
# - Long when price breaks above previous 24h high with volume > 1.5x 48-period average and rising ATR (volatility expansion)
# - Short when price breaks below previous 24h low with volume > 1.5x 48-period average and rising ATR
# - Uses rising ATR as volatility filter to avoid whipsaws in low volatility periods
# - Exits on opposite breakout (mean reversion tendency in ranging markets)
# - Position size 0.25 to balance risk and returns
# - Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag
# - Works in both bull and bear markets by capturing breakouts with volatility confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ATR for volatility filter (14-period)
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
        
        # Get previous 1d high/low for breakout levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        
        # Create arrays for alignment (constant values for the 1d period)
        high_array = np.full(len(df_1d), prev_high)
        low_array = np.full(len(df_1d), prev_low)
        
        # Align to 4h timeframe
        high_4h = align_htf_to_ltf(prices, df_1d, high_array)[i]
        low_4h = align_htf_to_ltf(prices, df_1d, low_array)[i]
        
        if position == 0:
            # Long: Break above previous 24h high with volume and volatility filter
            if (close[i] > high_4h and 
                volume[i] > vol_ma[i] * 1.5 and
                i >= 28 and atr[i] > atr[i-14]):
                position = 1
                signals[i] = position_size
            # Short: Break below previous 24h low with volume and volatility filter
            elif (close[i] < low_4h and 
                  volume[i] > vol_ma[i] * 1.5 and
                  i >= 28 and atr[i] > atr[i-14]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous 24h low (mean reversion in ranging markets)
            if close[i] < low_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous 24h high (mean reversion in ranging markets)
            if close[i] > high_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_DonchianBreakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0