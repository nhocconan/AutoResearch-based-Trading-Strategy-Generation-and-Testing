#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_1wTrend_Volume
# Hypothesis: On daily timeframe, price breaking Camarilla R1/S1 levels with weekly trend filter and volume confirmation captures strong directional moves. Weekly trend avoids counter-trend trades, volume reduces false breakouts. Designed for low frequency (10-25 trades/year) to minimize fee drift. Works in bull via R1 breakouts, in bear via S1 breakdowns.

name = "1d_Camarilla_R1S1_Breakout_1wTrend_Volume"
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
    
    # Daily Camarilla levels (based on previous day)
    # R1 = close + 1.12*(high - low)/12
    # S1 = close - 1.12*(high - low)/12
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    prev_close = close_series.shift(1)
    prev_high = high_series.shift(1)
    prev_low = low_series.shift(1)
    cam_range = prev_high - prev_low
    r1 = prev_close + 1.12 * cam_range / 12
    s1 = prev_close - 1.12 * cam_range / 12
    
    # Weekly trend: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation: 20-day average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
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
        volume_confirm = vol_ratio > 1.8
        
        if position == 0:
            # Enter long: break above R1 with weekly uptrend and volume
            if (close[i] > r1[i] and 
                trend_1w_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with weekly downtrend and volume
            elif (close[i] < s1[i] and 
                  trend_1w_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to S1 or trend fails
            if (close[i] < s1[i] or 
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to R1 or trend fails
            if (close[i] > r1[i] or 
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals