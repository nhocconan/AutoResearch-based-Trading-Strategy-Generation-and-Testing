#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1S1_Breakout_1wTrend_VolumeFilter
Hypothesis: Daily Camarilla pivot levels (R1/S1) breakouts with weekly trend filter (price > EMA50 weekly) and volume confirmation (current volume > 1.5x 20-day average) capture institutional breakout moves. Works in bull/bear markets by filtering with weekly trend. Targets 10-25 trades/year to minimize fee drag on 1d timeframe.
"""

name = "1d_Camarilla_Pivot_R1S1_Breakout_1wTrend_VolumeFilter"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend filter: EMA(50) on weekly close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Daily Camarilla pivot points (R1, S1) from previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar uses current
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for volume MA
        if position == 0:
            # LONG: Break above R1 with volume confirmation and weekly uptrend
            if (close[i] > r1[i] and 
                volume_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume confirmation and weekly downtrend
            elif (close[i] < s1[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or weekly trend reverses
            if (close[i] < r1[i]) or \
               (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or weekly trend reverses
            if (close[i] > s1[i]) or \
               (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals