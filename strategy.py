#!/usr/bin/env python3
# 1d_cci_breakout_1w_trend_volume
# Hypothesis: CCI(20) breakout on daily with weekly EMA trend filter and volume confirmation.
# Long when CCI crosses above +100 with price above weekly EMA50 and volume > 1.5x average.
# Short when CCI crosses below -100 with price below weekly EMA50 and volume > 1.5x average.
# Exit when CCI crosses back through zero.
# Designed to capture strong momentum moves with trend alignment in both bull and bear markets.
# Target: 30-100 total trades over 4 years (~7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_cci_breakout_1w_trend_volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate CCI(20) on daily data
    typical_price = (high + low + close) / 3
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - tp_ma.values) / (0.015 * tp_mad.values)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses back below zero
            if cci[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses back above zero
            if cci[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # CCI breakout entries
            if (cci[i] > 100) and (cci[i-1] <= 100) and (close[i] > ema_50_1w_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (cci[i] < -100) and (cci[i-1] >= -100) and (close[i] < ema_50_1w_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals