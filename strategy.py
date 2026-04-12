#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v2
# Uses weekly Camarilla pivot levels (H4/L4) as key support/resistance on daily chart.
# Long when price breaks above H4 with volume confirmation and weekly trend filter (weekly close > weekly SMA20).
# Short when price breaks below L4 with volume confirmation and weekly trend filter (weekly close < weekly SMA20).
# Exits when price returns to weekly pivot point (PP).
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and in ranging markets via mean reversion to pivot.
# Weekly trend filter reduces whipsaws in sideways markets.

name = "1d_1w_camarilla_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 weeks for SMA20
        return np.zeros(n)
    
    # Calculate weekly OHLC and SMA20 for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    sma20_1w = close_1w_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly trend: 1 = uptrend (close > SMA20), -1 = downtrend (close < SMA20), 0 = neutral
    weekly_trend = np.zeros(len(close_1w))
    weekly_trend[close_1w > sma20_1w] = 1
    weekly_trend[close_1w < sma20_1w] = -1
    
    # Calculate weekly Camarilla levels based on previous week's OHLC
    # Use previous week's data to avoid look-ahead
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev = np.roll(close_1w, 1)
    
    # Set first week's previous values to current week (will be filtered out by alignment)
    high_1w_prev[0] = high_1w[0]
    low_1w_prev[0] = low_1w[0]
    close_1w_prev[0] = close_1w[0]
    
    # Calculate pivot point and Camarilla levels for each week based on previous week
    pp = (high_1w_prev + low_1w_prev + close_1w_prev) / 3.0
    range_1w = high_1w_prev - low_1w_prev
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4 = pp + (1.1 / 2) * range_1w
    l4 = pp - (1.1 / 2) * range_1w
    
    # Align weekly levels to daily timeframe (weekly values update after weekly bar closes)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Volume confirmation: volume > 1.5 * 20-period average (daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(weekly_trend_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H4 with weekly uptrend
        if close[i] > h4_aligned[i] and weekly_trend_aligned[i] == 1 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L4 with weekly downtrend
        elif close[i] < l4_aligned[i] and weekly_trend_aligned[i] == -1 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to weekly pivot point (mean reversion)
        elif position == 1 and close[i] <= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals