#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_Volume_1dTrend
Hypothesis: TRIX(12) crossing zero line on 4h, filtered by 1d EMA34 trend and volume spike (2x median). TRIX filters whipsaws and captures momentum shifts. Works in bull (zero crosses up) and bear (zero crosses down). Target: 20-40 trades/year to avoid fee drag.
"""

name = "4h_TRIX_ZeroCross_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- TRIX(12) calculation (triple EMA) ---
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix_values = trix.fillna(0).values
    
    # --- Volume Filter: spike above 2x median of last 20 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 35  # for TRIX (12*3) + buffer
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Exit on TRIX zero cross in opposite direction
                if position == 1 and trix_values[i] < 0:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and trix_values[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema34_1d_aligned[i]
        trend_down = close_4h[i] < ema34_1d_aligned[i]
        
        # Volume filter: spike above 2x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if trix_values[i] > 0 and trix_values[i-1] <= 0 and trend_up and vol_ok:
                # Long: TRIX crosses up through zero + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif trix_values[i] < 0 and trix_values[i-1] >= 0 and trend_down and vol_ok:
                # Short: TRIX crosses down through zero + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Exit: TRIX crosses zero in opposite direction
            if position == 1:
                if trix_values[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if trix_values[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals