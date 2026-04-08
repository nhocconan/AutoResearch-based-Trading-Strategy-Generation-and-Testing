#!/usr/bin/env python3
# 4h_cci_breakout_12h_trend_volume_v1
# Hypothesis: CCI(20) extreme breakouts on 4h with 12h EMA trend filter and volume confirmation.
# Long when CCI crosses above +100 with uptrend (price > 12h EMA50) and volume > 1.3x average.
# Short when CCI crosses below -100 with downtrend (price < 12h EMA50) and volume > 1.3x average.
# Exit when CCI returns to neutral zone (-50 to 50).
# Designed to capture momentum bursts with trend alignment in both bull and bear markets.
# Target: 60-120 total trades over 4 years (~15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate CCI(20) on 4h data
    typical_price = (high + low + close) / 3
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(typical_price).rolling(window=20, min_periods=20).std().values
    # Avoid division by zero
    tp_std_safe = np.where(tp_std == 0, 1e-10, tp_std)
    cci = (typical_price - tp_mean) / (0.015 * tp_std_safe)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI returns to neutral zone (below 50)
            if cci[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI returns to neutral zone (above -50)
            if cci[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # CCI breakout entries: CCI > +100 (long) and CCI < -100 (short)
            if (cci[i] > 100) and (close[i] > ema_50_12h_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (cci[i] < -100) and (close[i] < ema_50_12h_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals