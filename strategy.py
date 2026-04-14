#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day ATR-based volatility breakout with volume confirmation
# - Buy when price breaks above (previous day close + 1.5 * daily ATR) with volume > 1.5x 24-period average
# - Sell when price breaks below (previous day close - 1.5 * daily ATR) with volume > 1.5x 24-period average
# - Uses daily ATR (20-period) for volatility measurement, updated only after daily bar close
# - Designed to capture volatility expansion moves in both bull and bear markets
# - Discrete position sizing (0.25) to minimize churn and manage drawdown
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR for volatility measurement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 4h volume filter: current volume > 1.5x 24-period average (1 day)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    # Create arrays for daily breakout levels
    upper_break_array = close_1d + 1.5 * atr_1d
    lower_break_array = close_1d - 1.5 * atr_1d
    
    # Align daily levels to 4h timeframe (will use previous day's values)
    upper_break_4h = align_htf_to_ltf(prices, df_1d, upper_break_array)
    lower_break_4h = align_htf_to_ltf(prices, df_1d, lower_break_array)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_1d[i]) or np.isnan(vol_ma[i]):
            continue
        
        if position == 0:
            # Long: Price breaks above upper level with volume confirmation
            if (close[i] > upper_break_4h[i] and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower level with volume confirmation
            elif (close[i] < lower_break_4h[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below lower level
            if close[i] < lower_break_4h[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above upper level
            if close[i] > upper_break_4h[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_ATRBreakout_Volume"
timeframe = "4h"
leverage = 1.0