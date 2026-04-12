#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_meanrev_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CCI(20) on daily data
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (tp_1d - sma_tp) / (0.015 * mad)
    
    # Align CCI to 4h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Volume filter - 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(cci_1d_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean-reversion signals with volume confirmation
        # Long: CCI < -100 (oversold) - contrarian long
        long_signal = cci_1d_aligned[i] < -100 and volume_ok[i]
        # Short: CCI > 100 (overbought) - contrarian short
        short_signal = cci_1d_aligned[i] > 100 and volume_ok[i]
        
        # Exit when CCI returns to neutral zone
        exit_long = cci_1d_aligned[i] > -100
        exit_short = cci_1d_aligned[i] < 100
        
        # Execute trades
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals