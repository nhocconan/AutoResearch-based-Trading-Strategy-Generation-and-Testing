#!/usr/bin/env python3
# 4h_trix_volume_regime_v1
# Hypothesis: TRIX momentum with volume spike confirmation and Choppiness regime filter on 4h.
# Long when TRIX crosses above zero with volume > 1.5x average and Choppiness > 61.8 (range).
# Short when TRIX crosses below zero with volume > 1.5x average and Choppiness > 61.8 (range).
# Exit when TRIX crosses back to zero or Choppiness < 38.2 (trend).
# Uses 1d EMA200 as trend filter: only long when price > EMA200, short when price < EMA200.
# Designed to capture mean-reversion in ranging markets and avoid trending whipsaws.
# Target: 80-160 total trades over 4 years (~20-40/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_volume_regime_v1"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate TRIX on 4h close (12-period)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 4h data (14-period)
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    
    atr_series = pd.Series(atr_list)
    sum_atr14 = atr_series.rolling(window=14, min_periods=14).sum()
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    chop = 100 * np.log10(sum_atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    chop_values = chop.fillna(50).values  # fill NaN with neutral value
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trix_values[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(chop_values[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR Choppiness < 38.2 (trending)
            if trix_values[i] < 0 or chop_values[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR Choppiness < 38.2 (trending)
            if trix_values[i] > 0 or chop_values[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            # Choppiness filter: > 61.8 (range-bound market)
            chop_ok = chop_values[i] > 61.8
            
            # TRIX cross above zero (bullish momentum)
            if (trix_values[i] > 0 and trix_values[i-1] <= 0) and \
               (close[i] > ema_200_1d_aligned[i]) and volume_ok and chop_ok:
                position = 1
                signals[i] = 0.25
            # TRIX cross below zero (bearish momentum)
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0) and \
                 (close[i] < ema_200_1d_aligned[i]) and volume_ok and chop_ok:
                position = -1
                signals[i] = -0.25
    
    return signals