# 6H_WEEKLY_PIVOT_BREAKOUT_1D_TREND
# Hypothesis: 6-hour breakouts above weekly R1 or below weekly S1 with 1-day EMA34 trend filter and volume confirmation.
# Weekly pivots provide weekly support/resistance structure, EMA34 filters daily trend direction, volume confirms breakout strength.
# Designed for low trade frequency (target: 12-37/year) with strong performance in both bull and bear markets.

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
    
    # Calculate 1-day EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 with proper smoothing
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Align 1-day EMA34 to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly pivots from previous week
    # Weekly Pivot: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H
    weekly_p = np.full(n, np.nan)
    weekly_r1 = np.full(n, np.nan)
    weekly_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous week's OHLC (approximate using daily data from 1d timeframe)
        # Since we don't have weekly data directly, we'll approximate using 5 daily periods
        if i >= 5:  # Approximate weekly lookback (5 days)
            week_high = np.max(high[i-5:i])
            week_low = np.min(low[i-5:i])
            week_close = close[i-1]
            weekly_p[i] = (week_high + week_low + week_close) / 3
            weekly_r1[i] = 2 * weekly_p[i] - week_low
            weekly_s1[i] = 2 * weekly_p[i] - week_high
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 5)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r1[i]) or np.isnan(weekly_s1[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume spike and 1d uptrend
            if (close[i] > weekly_r1[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike and 1d downtrend
            elif (close[i] < weekly_s1[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly S1 or 1d trend turns down
            if (close[i] < weekly_s1[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly R1 or 1d trend turns up
            if (close[i] > weekly_r1[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6H_WEEKLY_PIVOT_BREAKOUT_1D_TREND"
timeframe = "6h"
leverage = 1.0