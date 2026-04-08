#!/usr/bin/env python3
# 4h_cci_breakout_1d_trend_volume
# Hypothesis: CCI(20) breakout on 4h combined with 1d EMA trend filter and volume confirmation.
# Long when CCI crosses above +100 with uptrend (price > 1d EMA50) and volume > 1.5x average.
# Short when CCI crosses below -100 with downtrend (price < 1d EMA50) and volume > 1.5x average.
# Exit when CCI crosses back to zero.
# Designed to capture strong momentum breakouts with trend alignment in both bull and bear markets.
# Target: 80-180 total trades over 4 years (~20-45/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_1d_trend_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate CCI(20) on 4h data
    tp = (high + low + close) / 3
    tp_ma = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(tp).rolling(window=20, min_periods=20).std().values
    cci = (tp - tp_ma) / (0.015 * tp_std)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below zero
            if cci[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above zero
            if cci[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # CCI breakout entries
            if (cci[i] > 100) and (cci[i-1] <= 100) and (close[i] > ema_50_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (cci[i] < -100) and (cci[i-1] >= -100) and (close[i] < ema_50_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals