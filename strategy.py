#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_TrendFilter
Hypothesis: Uses TRIX (15) momentum on 12-hour timeframe with volume spike confirmation and 1-week EMA50 trend filter.
Goes long when TRIX crosses above zero with volume spike in uptrend, short when TRIX crosses below zero with volume spike in downtrend.
Designed to capture momentum shifts with filtered false signals via volume and trend alignment.
Targets 12-37 trades per year to minimize fee dust while capturing meaningful momentum shifts.
Works in bull/bear via trend filter and momentum reversal logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate TRIX (15) on 12h close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # percent change
    trix = np.concatenate([[np.nan], trix_raw])  # align with original length
    
    # Calculate volume spike (>2.0x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1-week EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # TRIX zero-cross signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Entry logic:
        # Long: TRIX crosses above zero with volume spike in uptrend
        long_entry = vol_confirm and trend_up and trix_cross_up
        
        # Short: TRIX crosses below zero with volume spike in downtrend
        short_entry = vol_confirm and trend_down and trix_cross_down
        
        # Exit logic: Opposite TRIX cross or trend reversal
        long_exit = trix_cross_down or not trend_up
        short_exit = trix_cross_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_TRIX_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0