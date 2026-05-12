#!/usr/bin/env python3
# 1h Volume Breakout with 4h Trend and Daily Volume Filter
# Hypothesis: Volume breakouts on 1h capture momentum bursts. 4h EMA50 provides trend filter (only trade in trend direction).
# Daily volume average filter ensures we only trade on high-volume days, reducing false signals in low-activity periods.
# Works in bull markets (breakouts continue trends) and bear markets (breakouts catch reversals or strong bounces).
# Designed for low trade frequency (~15-35/year) with clear entry/exit rules.

name = "1h_VolumeBreakout_4hTrend_DailyVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Data for EMA Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === Daily Data for Volume Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # === 1h Indicators ===
    # 20-period volume moving average for breakout detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    # 2-period high/low for breakout levels
    high_2 = pd.Series(high).rolling(window=2, min_periods=2).max().values
    low_2 = pd.Series(low).rolling(window=2, min_periods=2).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_2[i]) or np.isnan(low_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Volume breakout above 2-period high + above 4h EMA50 + volume > 1.5x daily avg
            if (close[i] > high_2[i] and 
                close[i] > ema_50_4h_aligned[i] and
                volume[i] > (vol_ma[i] * 2.0) and  # intraday volume spike
                volume[i] > (vol_avg_1d_aligned[i] * 1.5)):  # above daily average
                signals[i] = 0.20
                position = 1
            # SHORT: Volume breakout below 2-period low + below 4h EMA50 + volume > 1.5x daily avg
            elif (close[i] < low_2[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  volume[i] > (vol_ma[i] * 2.0) and  # intraday volume spike
                  volume[i] > (vol_avg_1d_aligned[i] * 1.5)):  # above daily average
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below 2-period low or closes below 4h EMA50
            if close[i] < low_2[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 2-period high or closes above 4h EMA50
            if close[i] > high_2[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals