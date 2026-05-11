#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Uses daily trend (EMA34) to filter direction and Camarilla R1/S1 levels from the prior day for breakout entries.
Enters long when price breaks above R1 in a daily uptrend with volume confirmation.
Enters short when price breaks below S1 in a daily downtrend with volume confirmation.
Exits when price returns to the prior day's pivot point (PP) or reverses intraday.
Designed to work in both bull and bear markets by following daily EMA trend.
Targets 25-40 trades/year via strict breakout conditions and trend filter.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the given period"""
    range_val = high - low
    pp = (high + low + close) / 3
    r1 = pp + (range_val * 1.1 / 12)
    s1 = pp - (range_val * 1.1 / 12)
    return pp, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Trend Filter (EMA34) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema_34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend = np.where(daily_close > ema_34, 1, -1)
    
    # Align daily trend to 4h
    daily_trend_4h = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # --- Daily Camarilla Levels (from prior day) ---
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_for_cam = df_1d['close'].values
    
    _, r1, s1 = calculate_camarilla(daily_high, daily_low, daily_close_for_cam)
    pp_val, _, _ = calculate_camarilla(daily_high, daily_low, daily_close_for_cam)
    
    # Align Camarilla levels to 4h (shifted by 1 day to avoid look-ahead)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    pp_4h = align_htf_to_ltf(prices, df_1d, pp_val)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_trend_4h[i]) or np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or np.isnan(pp_4h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        # Daily trend direction
        trend = daily_trend_4h[i]
        
        if position == 0:
            # Long: daily uptrend + price breaks above R1 + volume
            if (trend == 1 and 
                close[i] > r1_4h[i] and 
                close[i-1] <= r1_4h[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend + price breaks below S1 + volume
            elif (trend == -1 and 
                  close[i] < s1_4h[i] and 
                  close[i-1] >= s1_4h[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to PP or daily trend turns down
                if (close[i] <= pp_4h[i] or trend == -1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to PP or daily trend turns up
                if (close[i] >= pp_4h[i] or trend == 1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals