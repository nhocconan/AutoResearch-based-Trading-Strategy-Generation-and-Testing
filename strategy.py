#!/usr/bin/env python3
# 1D_WeeklyTrend_Breakout_55
# Hypothesis: Buy weekly trend pullbacks to 55-period EMA on daily chart with volume confirmation.
# Long when: weekly uptrend + price pulls back to daily EMA55 + volume > 1.5x average.
# Short when: weekly downtrend + price rallies to daily EMA55 + volume > 1.5x average.
# Uses volume confirmation to filter false breakouts and weekly trend to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol.

name = "1D_WeeklyTrend_Breakout_55"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA55 for dynamic support/resistance
    close_s = pd.Series(close)
    ema55 = close_s.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Volume average (20-period)
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly trend: price above/below 21-period EMA
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = close_1w > ema21_1w
    weekly_downtrend = close_1w < ema21_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema55[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + price near EMA55 + volume confirmation
            if weekly_up and volume_confirm:
                if close[i] <= ema55[i] * 1.01 and close[i] >= ema55[i] * 0.99:
                    signals[i] = 0.25
                    position = 1
            # Enter short: weekly downtrend + price near EMA55 + volume confirmation
            elif weekly_down and volume_confirm:
                if close[i] >= ema55[i] * 0.99 and close[i] <= ema55[i] * 1.01:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: weekly trend changes or price moves away from EMA55
            if not weekly_up or close[i] > ema55[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: weekly trend changes or price moves away from EMA55
            if not weekly_down or close[i] < ema55[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals