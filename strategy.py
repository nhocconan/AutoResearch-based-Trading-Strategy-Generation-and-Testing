#!/usr/bin/env python3
# 12h_trix_volume_regime_v1
# Hypothesis: TRIX momentum on 12h with volume confirmation and weekly trend filter.
# Long when TRIX > 0, volume > 1.5x average, and price above weekly EMA(50).
# Short when TRIX < 0, volume > 1.5x average, and price below weekly EMA(50).
# Uses 1w EMA(50) as trend filter to avoid counter-trend trades. Target: 15-25 trades/year.
# Works in bull/bear: trend filter ensures alignment with higher timeframe momentum.

name = "12h_trix_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX calculation (1-period ROC of triple EMA)
    def calculate_trix(prices, period=15):
        ema1 = pd.Series(prices).ewm(span=period, adjust=False, min_periods=period).mean()
        ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
        ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
        # 1-period ROC of ema3
        trix = np.zeros_like(prices)
        trix[period:] = (ema3.values[period:] - ema3.values[:-period]) / ema3.values[:-period] * 100
        return trix
    
    trix = calculate_trix(close, 15)
    
    # Weekly EMA trend filter (50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(15, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if TRIX turns negative or trend fails
            if trix[i] <= 0 or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if TRIX turns positive or trend fails
            if trix[i] >= 0 or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TRIX positive with uptrend and volume
            if trix[i] > 0 and close[i] > ema_1w_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: TRIX negative with downtrend and volume
            elif trix[i] < 0 and close[i] < ema_1w_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals