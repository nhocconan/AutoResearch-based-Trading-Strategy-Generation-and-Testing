#!/usr/bin/env python3
"""
4h_HTF_Pullback_Bounce_v1
Hypothesis: In strong trends (weekly EMA34 slope > 0), price pulls back to the daily EMA34
and bounces with volume confirmation. Works in bull (long on pullbacks) and bear
(short on pullbacks to resistance). Uses 4h for timing, daily/weekly for trend/filter.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA (for EMA34 pullback) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA34
    if len(close_1d) >= 34:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_34_1d = np.full_like(close_1d, np.nan)
    
    # Align daily EMA34 to 4h
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === WEEKLY DATA (for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 and its slope (trend strength)
    if len(close_1w) >= 34:
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
        # Slope: positive = uptrend, negative = downtrend
        ema_slope_1w = np.diff(ema_34_1w, prepend=ema_34_1w[0])
    else:
        ema_34_1w = np.full_like(close_1w, np.nan)
        ema_slope_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA and slope to 4h
    ema_34_1w_4h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_slope_1w_4h = align_htf_to_ltf(prices, df_1w, ema_slope_1w)
    
    # === VOLUME FILTER (4h) ===
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data is not available
        if (np.isnan(ema_34_4h[i]) or np.isnan(ema_34_1w_4h[i]) or 
            np.isnan(ema_slope_1w_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: weekly EMA34 slope > 0 for long, < 0 for short
        uptrend = ema_slope_1w_4h[i] > 0
        downtrend = ema_slope_1w_4h[i] < 0
        
        if position == 0:
            # Long: pullback to daily EMA34 support in uptrend with volume
            if close[i] <= ema_34_4h[i] * 1.005 and close[i] >= ema_34_4h[i] * 0.995 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: pullback to daily EMA34 resistance in downtrend with volume
            elif close[i] >= ema_34_4h[i] * 0.995 and close[i] <= ema_34_4h[i] * 1.005 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below daily EMA34 OR trend turns down
            if close[i] < ema_34_4h[i] * 0.99 or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily EMA34 OR trend turns up
            if close[i] > ema_34_1w_4h[i] * 1.01 or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Pullback_Bounce_v1"
timeframe = "4h"
leverage = 1.0