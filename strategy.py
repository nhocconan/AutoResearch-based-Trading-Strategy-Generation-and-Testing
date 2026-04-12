#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_filtered_v2
# Uses daily Camarilla pivot levels (H4/L4) for trend direction.
# 1h timeframe for precise entry timing on pullbacks to 4h EMA.
# Volume confirmation and session filter (08-20 UTC) to reduce noise.
# In bull markets: long when price pulls back to 4h EMA then breaks above prior hour high.
# In bear markets: short when price pulls back to 4h EMA then breaks below prior hour low.
# Daily trend filter: only trade in direction of daily close vs 200 EMA.
# Target: 20-30 trades/year per symbol for low friction and high edge.

name = "1h_4h_1d_camarilla_filtered_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_uptrend = close_1d > ema200_1d
    
    # Previous day's Camarilla levels (H4/L4)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align daily levels to 1h
    h4_level = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    
    # Get 4h data for EMA21
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # start after warmup
        # Skip if not in trading session
        if not session_filter[i]:
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or np.isnan(ema21_4h_aligned[i]) or np.isnan(daily_uptrend_aligned[i]):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trading bias from daily trend
        is_uptrend = daily_uptrend_aligned[i] > 0.5
        
        # Long setup: uptrend + price near 4h EMA21 + break above prior hour high
        if is_uptrend and close[i] > ema21_4h_aligned[i] * 0.995 and close[i] < ema21_4h_aligned[i] * 1.005:
            if i > 0 and close[i] > high[i-1] and position != 1:
                position = 1
                signals[i] = 0.20
        
        # Short setup: downtrend + price near 4h EMA21 + break below prior hour low
        elif not is_uptrend and close[i] > ema21_4h_aligned[i] * 0.995 and close[i] < ema21_4h_aligned[i] * 1.005:
            if i > 0 and close[i] < low[i-1] and position != -1:
                position = -1
                signals[i] = -0.20
        
        # Exit conditions: contrary break of prior hour extreme
        elif position == 1 and i > 0 and close[i] < low[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and i > 0 and close[i] > high[i-1]:
            position = 0
            signals[i] = 0.0
        
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals