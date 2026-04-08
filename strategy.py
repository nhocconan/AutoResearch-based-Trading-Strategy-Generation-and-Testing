#!/usr/bin/env python3
# 6h_trix_volume_regime_v1
# Hypothesis: On 6h timeframe, capture momentum reversals using TRIX (triple-smoothed EMA) combined with volume surge and weekly trend filter.
# TRIX crossing zero indicates momentum shift. Volume > 1.5x 20-period average confirms conviction.
# Weekly trend filter (price > EMA50) ensures alignment with higher timeframe direction.
# Works in bull/bear by following momentum with volume confirmation. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_volume_regime_v1"
timeframe = "6h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # TRIX (15,9,9) - triple smoothed EMA
    # First EMA
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    # TRIX = % change of triple smoothed EMA
    trix = np.zeros(n)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trix[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR volume drops below average
            if (trix[i] < 0 and trix[i-1] >= 0) or (volume[i] < vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR volume drops below average
            if (trix[i] > 0 and trix[i-1] <= 0) or (volume[i] < vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry conditions: TRIX crosses zero with volume surge and weekly trend alignment
            # Long: TRIX crosses above zero, volume > 1.5x average, price above weekly EMA50
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: TRIX crosses below zero, volume > 1.5x average, price below weekly EMA50
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals