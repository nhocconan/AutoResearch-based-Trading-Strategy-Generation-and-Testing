#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_Volume_Spike_Trend
# Hypothesis: TRIX zero cross indicates momentum shift. Combined with volume spike and 1d trend filter.
# Long: TRIX crosses above zero + volume spike + 1d uptrend. Short: TRIX crosses below zero + volume spike + 1d downtrend.
# Uses TRIX(12) for momentum, volume confirmation for conviction, and 1d EMA34 for trend filter.
# Designed for low-frequency, high-conviction trades in both bull and bear markets.

name = "4h_TRIX_ZeroCross_Volume_Spike_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # TRIX calculation: Triple EMA of ROC
    # ROC(1) = (close - close.shift(1)) / close.shift(1)
    close_series = pd.Series(close)
    roc = close_series.pct_change(periods=1)
    # EMA1 of ROC
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    # EMA2 of EMA1
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    # EMA3 of EMA2 (TRIX)
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 * 100).values  # Scale for readability
    
    # TRIX zero cross signals
    trix_above_zero = trix > 0
    trix_below_zero = trix < 0
    trix_cross_up = (trix > 0) & (np.concatenate([[False], trix[:-1] <= 0]))  # Previous <=0, current >0
    trix_cross_down = (trix < 0) & (np.concatenate([[False], trix[:-1] >= 0]))  # Previous >=0, current <0
    
    # Volume spike: volume > 2.0 * 20-period average (high threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend conditions
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: TRIX crosses up + volume spike + uptrend
            if trix_cross_up[i] and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses down + volume spike + downtrend
            elif trix_cross_down[i] and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses down OR trend reversal
            if trix_cross_down[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses up OR trend reversal
            if trix_cross_up[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals