#!/usr/bin/env python3
# 12h_Weekly_Pullback_1dTrend_Volume
# Hypothesis: In higher timeframe (12h), pullbacks to 1-day EMA34 during weekly trend (aligned with 1-week EMA50) with volume confirmation capture mean-reversion moves in both bull and bear markets. Weekly trend filter avoids counter-trend trades, while 1d EMA34 acts as dynamic support/resistance. Volume spike ensures institutional participation. Designed for low trade frequency (~15-30/year) to minimize fee drag on 12h chart.

name = "12h_Weekly_Pullback_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 (dynamic support/resistance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1w EMA50 trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1d EMA34 to 12h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume spike: current > 2.0 * 30-period average (12h chart)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(trend_1w_up_aligned[i]) or 
            np.isnan(trend_1w_down_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: pullback to 1d EMA34 support during 1w uptrend with volume spike
            if (low[i] <= ema34_1d_aligned[i] and 
                trend_1w_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: pullback to 1d EMA34 resistance during 1w downtrend with volume spike
            elif (high[i] >= ema34_1d_aligned[i] and 
                  trend_1w_down_aligned[i] > 0.5 and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price moves above 1d EMA34 (taking profit) or trend fails
            if (close[i] > ema34_1d_aligned[i] or 
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price moves below 1d EMA34 (taking profit) or trend fails
            if (close[i] < ema34_1d_aligned[i] or 
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals