#!/usr/bin/env python3
"""
4h_1d_Camarilla_Trend_Breakout_v1
Hypothesis: Buy breakouts above Camarilla H3 when 1d EMA50 > EMA200 (uptrend) with volume confirmation.
Sell breakdowns below L3 when 1d EMA50 < EMA200 (downtrend) with volume confirmation.
Exit at opposite H3/L3 levels. Uses daily trend filter to avoid counter-trend trades.
Designed for low trade frequency (<40/year) by requiring trend alignment and volume confirmation.
Works in bull/bear via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Trend_Breakout_v1"
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
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === CAMARILLA LEVELS (using previous day's close) ===
    # Shift close by 1 to get previous day's close
    close_prev = np.concatenate([[close_1d[0]], close_1d[:-1]])
    range_1d = high_1d - low_1d
    
    # Calculate levels
    h3 = close_prev + (range_1d * 1.1 / 4)
    l3 = close_prev - (range_1d * 1.1 / 4)
    h4 = close_prev + (range_1d * 1.1)
    l4 = close_prev - (range_1d * 1.1)
    
    # === DAILY TREND FILTER: EMA50 > EMA200 for uptrend ===
    close_series = pd.Series(close_1d)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    # uptrend = 1 when EMA50 > EMA200, downtrend = -1 when EMA50 < EMA200, 0 otherwise
    daily_trend = np.where(ema_50 > ema_200, 1, np.where(ema_50 < ema_200, -1, 0))
    
    # === VOLUME AVERAGE (20-period) ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    # === ALIGN TO 4H TIMEFRAME ===
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(daily_trend_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Get current trend (-1, 0, 1)
        trend = int(daily_trend_aligned[i])
        
        # Entry conditions: trend-aligned breakouts with volume
        long_setup = (trend == 1) and (close[i] > h3_aligned[i]) and vol_confirm
        short_setup = (trend == -1) and (close[i] < l3_aligned[i]) and vol_confirm
        
        # Exit conditions: reverse signal or opposite level touch
        exit_long = (trend == -1) or (close[i] < l3_aligned[i])
        exit_short = (trend == 1) or (close[i] > h3_aligned[i])
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals