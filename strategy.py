#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_cci_reversal_v1
# Uses weekly Commodity Channel Index (CCI) to detect overbought/oversold conditions
# combined with daily price action for entry timing. CCI > 100 = overbought (short),
# CCI < -100 = oversold (long). Works in both bull and bear markets by capturing
# mean reversals at extremes. Low trade frequency expected due to strict CCI thresholds.
name = "1d_1w_cci_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for CCI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly CCI(20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Simple moving average of typical price
    tp_ma = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    
    # Mean deviation
    tp_dev = np.abs(tp_1w - tp_ma)
    md = pd.Series(tp_dev).rolling(window=20, min_periods=20).mean().values
    
    # CCI calculation
    cci = (tp_1w - tp_ma) / (0.015 * md)
    
    # Align CCI to daily timeframe (with proper delay for weekly bar close)
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after CCI warmup
        # Skip if CCI not ready
        if np.isnan(cci_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        cci_val = cci_aligned[i]
        
        # Entry signals based on CCI extremes
        long_signal = cci_val < -100  # Oversold
        short_signal = cci_val > 100  # Overbought
        
        # Exit conditions
        exit_long = cci_val > -50  # Exit long when CCI rises above -50
        exit_short = cci_val < 50  # Exit short when CCI falls below 50
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals