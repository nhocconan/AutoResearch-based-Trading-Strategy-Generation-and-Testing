#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_WeeklyPivot_Touch_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and daily data for trend filter
    df_w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly pivot calculation (previous week's OHLC)
    prev_week_high = np.roll(df_w['high'].values, 1)
    prev_week_low = np.roll(df_w['low'].values, 1)
    prev_week_close = np.roll(df_w['close'].values, 1)
    # Handle first value
    prev_week_high[0] = df_w['high'].values[0]
    prev_week_low[0] = df_w['low'].values[0]
    prev_week_close[0] = df_w['close'].values[0]
    
    # Weekly pivot and support/resistance levels
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    # Weekly support 1 and resistance 1 (key levels)
    ws1 = weekly_pivot - weekly_range
    wr1 = weekly_pivot + weekly_range
    
    # Align weekly levels to 4h timeframe
    ws1_4h = align_htf_to_ltf(prices, df_w, ws1)
    wr1_4h = align_htf_to_ltf(prices, df_w, wr1)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_1d > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ws1_4h[i]) or np.isnan(wr1_4h[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price touches or goes below WS1 with volume spike and 1d uptrend
            long_cond = (low[i] <= ws1_4h[i] and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price touches or goes above WR1 with volume spike and 1d downtrend
            short_cond = (high[i] >= wr1_4h[i] and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back above weekly pivot (mean reversion)
            if close[i] > weekly_pivot[i] if hasattr(weekly_pivot, '__getitem__') else weekly_pivot:
                # Need to get the current weekly pivot value
                wp_current = align_htf_to_ltf(prices, df_w, weekly_pivot)[i]
                if close[i] > wp_current:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                wp_current = align_htf_to_ltf(prices, df_w, weekly_pivot)[i]
                if close[i] > wp_current:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back below weekly pivot
            wp_current = align_htf_to_ltf(prices, df_w, weekly_pivot)[i]
            if close[i] < wp_current:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot touch strategy with 1d trend filter and volume spike confirmation on 4h timeframe.
# Enters long when price touches or goes below weekly support 1 (WS1) with volume spike and 1d uptrend.
# Enters short when price touches or goes above weekly resistance 1 (WR1) with volume spike and 1d downtrend.
# Exits when price crosses back above/below the weekly pivot level.
# Uses weekly pivot levels for institutional significance, volume spike for confirmation,
# and 1d EMA34 trend filter to align with higher timeframe momentum.
# Designed to work in both bull and bear markets by trading mean reversion from extreme weekly levels.
# Uses 20-period volume MA with 2.0x threshold for balanced frequency. Targets 20-40 trades/year.