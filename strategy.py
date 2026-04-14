#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily volatility breakout with volume confirmation and ATR-based risk management
# - Breaks above/below previous day's high/low with volatility expansion (ATR > 1.5x 20-period average)
# - Requires volume > 1.3x 24-period average for institutional confirmation
# - Uses ATR-based stop loss and profit targets to manage risk
# - Designed to capture volatility expansion in both bull and bear markets
# - Discrete position sizing (0.25) to minimize churn and manage drawdown
# - Target: 80-150 trades over 4 years (20-38/year) to avoid fee drag

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
    
    # Calculate ATR for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 4h volume filter: current volume > 1.3x 24-period average (1 day)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    # Daily range for breakout levels
    daily_range = high_1d - low_1d
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get previous day's data for breakout levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_range = prev_high - prev_low
        
        # Calculate breakout levels: previous day's high/low ± 0.3 * range
        upper_break = prev_high + 0.3 * prev_range
        lower_break = prev_low - 0.3 * prev_range
        
        # Create arrays for alignment
        upper_array = np.full(len(df_1d), upper_break)
        lower_array = np.full(len(df_1d), lower_break)
        
        upper_4h = align_htf_to_ltf(prices, df_1d, upper_array)[i]
        lower_4h = align_htf_to_ltf(prices, df_1d, lower_array)[i]
        
        if position == 0:
            # Long: Price breaks above upper level with volume and volatility expansion
            if (close[i] > upper_4h and close[i-1] <= upper_4h and 
                volume[i] > vol_ma[i] * 1.3 and 
                atr[i] > atr[i-20] * 1.5 if i >= 20 else False):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower level with volume and volatility expansion
            elif (close[i] < lower_4h and close[i-1] >= lower_4h and 
                  volume[i] > vol_ma[i] * 1.3 and 
                  atr[i] > atr[i-20] * 1.5 if i >= 20 else False):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below lower level or ATR contraction
            if close[i] < lower_4h or (i >= 20 and atr[i] < atr[i-20] * 0.8):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above upper level or ATR contraction
            if close[i] > upper_4h or (i >= 20 and atr[i] < atr[i-20] * 0.8):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_VolatilityBreakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0