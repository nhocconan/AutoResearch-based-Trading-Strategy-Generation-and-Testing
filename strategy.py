#!/usr/bin/env python3
# 6h_1w_1d_trix_volume_v1
# Strategy: 6h TRIX (1-period ROC of triple-smoothed EMA) with volume confirmation and weekly trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: TRIX filters noise and identifies momentum. In bull markets, TRIX > 0 with rising signal line and volume confirmation. In bear markets, TRIX < 0 with falling signal line and volume confirmation. Weekly trend filter ensures alignment with higher timeframe trend. Targets 15-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_trix_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily EMA(200) for regime filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # TRIX calculation (15-period triple EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change(1) * 100  # 1-period ROC
    trix_values = trix.values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after TRIX and signal line warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(trix_values[i]) or np.isnan(trix_signal[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: price above/below 1d EMA200
        bull_regime = close[i] > ema_200_1d_aligned[i]
        bear_regime = close[i] < ema_200_1d_aligned[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: TRIX momentum + volume + trend/regime alignment
        if (trix_values[i] > 0 and trix_values[i] > trix_signal[i] and  # TRIX positive and above signal
            vol_confirm[i] and uptrend and bull_regime and position != 1):
            position = 1
            signals[i] = 0.25
        elif (trix_values[i] < 0 and trix_values[i] < trix_signal[i] and  # TRIX negative and below signal
              vol_confirm[i] and downtrend and bear_regime and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TRIX crossover or regime change
        elif position == 1 and (trix_values[i] <= trix_signal[i] or not bull_regime or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix_values[i] >= trix_signal[i] or not bear_regime or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals