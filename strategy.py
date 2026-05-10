#!/usr/bin/env python3
# 6h_Donchian_20_WeeklyTrend_DailyVol
# Hypothesis: Donchian(20) breakout on 6h with weekly trend filter and daily volume confirmation.
# Works in bull (breakouts above upper band) and bear (breakouts below lower band) by
# aligning with weekly trend direction. Volume spike confirms breakout strength.
# Uses 6h primary, 1w trend, 1d volume - avoids overtrading with strict 3-condition entry.

name = "6h_Donchian_20_WeeklyTrend_DailyVol"
timeframe = "6h"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to 6h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_series_1d = pd.Series(volume_1d)
    vol_ma_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily volume ratio (current 6h volume vs daily average)
        # Scale 6h volume to daily equivalent: 6h volume * 4 = approximate daily volume
        vol_6h = volume[i]
        vol_daily_equiv = vol_ma_1d_aligned[i] / 4.0  # daily average / 4 = 6h average
        vol_ratio = vol_6h / vol_daily_equiv if vol_daily_equiv > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above upper band with weekly uptrend and volume spike
            if (close[i] > upper[i] and 
                trend_1w_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with weekly downtrend and volume spike
            elif (close[i] < lower[i] and 
                  trend_1w_down_aligned[i] > 0.5 and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below lower band or trend fails
            if (close[i] < lower[i] or 
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above upper band or trend fails
            if (close[i] > upper[i] or 
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals