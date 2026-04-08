#!/usr/bin/env python3
# 1d_trix_volume_sr_1w_trend
# Hypothesis: TRIX momentum on 1d filtered by 1w EMA trend and volume confirmation. 
# Long when TRIX crosses above zero with uptrend (price > 1w EMA40) and volume > 1.5x average.
# Short when TRIX crosses below zero with downtrend (price < 1w EMA40) and volume > 1.5x average.
# Designed to capture momentum shifts in both bull and bear markets with low trade frequency.
# Target: 10-25 trades/year (~40-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_trix_volume_sr_1w_trend"
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
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Calculate TRIX (15-period EMA applied 3 times)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trix[i]) or np.isnan(ema_40_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR trend turns against us
            if (trix[i] < 0) or (close[i] < ema_40_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR trend turns against us
            if (trix[i] > 0) or (close[i] > ema_40_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: TRIX crosses above zero with uptrend and volume confirmation
            if (trix[i] > 0) and (close[i] > ema_40_1w_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: TRIX crosses below zero with downtrend and volume confirmation
            elif (trix[i] < 0) and (close[i] < ema_40_1w_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals