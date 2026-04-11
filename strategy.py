#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate CCI (Commodity Channel Index) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # 20-period SMA of typical price
    tp_ma_20 = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    
    # Mean deviation
    tp_dev = np.abs(tp_1d - tp_ma_20)
    tp_dev_ma_20 = pd.Series(tp_dev).rolling(window=20, min_periods=20).mean().values
    
    # CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    cci_1d = (tp_1d - tp_ma_20) / (0.015 * tp_dev_ma_20 + 1e-10)
    
    # Align daily CCI to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Volume confirmation: 6h volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_30[i]
        
        # Mean reversion conditions
        cci_oversold = cci_aligned[i] < -150  # Deep oversold
        cci_overbought = cci_aligned[i] > 150  # Deep overbought
        
        # Exit conditions: return to neutral zone
        cci_exit_long = cci_aligned[i] > -50   # Exit long when CCI rises above -50
        cci_exit_short = cci_aligned[i] < 50   # Exit short when CCI falls below 50
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Deep oversold with volume confirmation
        if cci_oversold and vol_confirm:
            enter_long = True
        
        # Short: Deep overbought with volume confirmation
        if cci_overbought and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = cci_exit_long
        exit_short = cci_exit_short
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h CCI mean reversion strategy using daily CCI extremes with volume confirmation.
# Enters long when daily CCI < -150 (deep oversold) with volume > 1.5x 30-period average.
# Enters short when daily CCI > 150 (deep overbought) with volume > 1.5x 30-period average.
# Exits when CCI returns to neutral zone (> -50 for longs, < 50 for shorts).
# Uses daily timeframe for CCI calculation to avoid noise and capture stronger mean reversion signals.
# 6h timeframe provides sufficient frequency while avoiding excessive trading.
# Works in both bull and bear markets by exploiting overbought/oversold conditions that occur in all regimes.
# Volume confirmation ensures entries occur during periods of heightened interest, reducing false signals.
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag.
# Position size set to 0.25 to balance risk and reward while managing drawdown.