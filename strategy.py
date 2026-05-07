#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_WeeklyTrend_Filter
# Hypothesis: On daily chart, enter long when price breaks above Camarilla R1 with weekly trend filter (price above 200-week EMA) and volume confirmation.
# Enter short when price breaks below Camarilla S1 with weekly trend filter (price below 200-week EMA) and volume confirmation.
# Uses weekly EMA trend filter to avoid counter-trend trades in strong trends, improving win rate in both bull and bear markets.
# Volume confirmation reduces false breakouts. Target low trade frequency (~10-20/year) to minimize fee drag.
# Weekly trend filter uses 1h data as proxy for weekly trend (since 1w data not available, using 1d EMA200 as weekly trend proxy).
# Strategy uses Camarilla levels calculated from previous day's high, low, close.
# Exit when price returns to Camarilla pivot level (mean reversion) or opposite Camarilla level is breached with volume.

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
    
    # Calculate Camarilla levels for each day using previous day's HLC
    # Camarilla: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # H2 = Close + 0.6 * (High - Low)
    # H1 = Close + 0.318 * (High - Low)
    # L1 = Close - 0.318 * (High - Low)
    # L2 = Close - 0.6 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    # We use R1 = H1, S1 = L1
    
    # Shift to get previous day's values
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1] if n > 1 else high[0]  # handle first bar
    prev_low[0] = prev_low[1] if n > 1 else low[0]
    prev_close[0] = prev_close[1] if n > 1 else close[0]
    
    # Calculate Camarilla levels based on previous day
    hl_range = prev_high - prev_low
    camarilla_h1 = prev_close + 0.318 * hl_range  # R1
    camarilla_l1 = prev_close - 0.318 * hl_range  # S1
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Weekly trend filter: use 200-day EMA as proxy for weekly trend
    # (Since 1w data not available in standard timeframes, using 200-day EMA)
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup period
        # Skip if any critical value is NaN
        if (np.isnan(ema200[i]) or np.isnan(camarilla_h1[i]) or np.isnan(camarilla_l1[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(vol_ma20[i]) or vol_ma20[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + price above 200-day EMA (uptrend) + volume spike
            if close[i] > camarilla_h1[i] and close[i] > ema200[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + price below 200-day EMA (downtrend) + volume spike
            elif close[i] < camarilla_l1[i] and close[i] < ema200[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long:
            # 1. Price returns to Camarilla pivot (mean reversion target)
            # 2. Price breaks below Camarilla S1 with volume (trend reversal)
            if close[i] <= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < camarilla_l1[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short:
            # 1. Price returns to Camarilla pivot (mean reversion target)
            # 2. Price breaks above Camarilla R1 with volume (trend reversal)
            if close[i] >= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > camarilla_h1[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0