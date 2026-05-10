#!/usr/bin/env python3
# 1d_Camarilla_Pivot_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: On daily timeframe, Camarilla R1/S1 breakout with weekly trend filter and volume confirmation provides high-probability entries. Weekly trend avoids counter-trend trades, volume reduces false signals. Camarilla levels act as natural support/resistance. Designed for low frequency (~10-25 trades/year) to minimize fee drag while working in both bull and bear markets.

name = "1d_Camarilla_Pivot_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily Camarilla levels (using previous day's OHLC)
    shift_high = np.roll(high, 1)
    shift_low = np.roll(low, 1)
    shift_close = np.roll(close, 1)
    shift_high[0] = high[0]
    shift_low[0] = low[0]
    shift_close[0] = close[0]
    
    # Camarilla calculations
    cam_high = shift_high
    cam_low = shift_low
    cam_close = shift_close
    cam_range = cam_high - cam_low
    
    # Camarilla R1, S1 levels
    r1 = cam_close + (cam_range * 1.1 / 12)
    s1 = cam_close - (cam_range * 1.1 / 12)
    
    # Weekly trend: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation: 20-day average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above R1 with weekly uptrend and volume
            if (close[i] > r1[i] and 
                trend_1w_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with weekly downtrend and volume
            elif (close[i] < s1[i] and 
                  trend_1w_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to Camarilla H5/L5 midpoint or trend fails
            h5 = cam_close + (cam_range * 1.1 / 2)
            l5 = cam_close - (cam_range * 1.1 / 2)
            midpoint = (h5 + l5) / 2
            
            if (close[i] < midpoint[i] or 
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to Camarilla H5/L5 midpoint or trend fails
            h5 = cam_close + (cam_range * 1.1 / 2)
            l5 = cam_close - (cam_range * 1.1 / 2)
            midpoint = (h5 + l5) / 2
            
            if (close[i] > midpoint[i] or 
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals