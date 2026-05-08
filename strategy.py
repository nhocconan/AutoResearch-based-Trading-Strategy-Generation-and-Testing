# 6H_WeeklyPivot_R3S4_Breakout_1dTrend_Volume
# Hypothesis: Use weekly pivot levels (R3/S4) for breakout entries with 1d trend filter and volume confirmation.
# Weekly pivots provide strong support/resistance; breaks indicate momentum. 1d trend filter avoids counter-trend trades.
# Volume confirmation reduces false breakouts. Designed for 6s timeframe to capture multi-day moves with infrequent trades.
# Works in bull (breakouts catch trends) and bear (fades at strong levels) via directional filtering.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6H_WeeklyPivot_R3S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly Range = H - L
    range_1w = high_1w - low_1w
    # Resistance levels
    r3_1w = pivot_1w + (range_1w * 1.1)  # R3 = Pivot + 1.1 * Range
    s4_1w = pivot_1w - (range_1w * 1.6)  # S4 = Pivot - 1.6 * Range
    
    # Align weekly levels to 6h timeframe (with 1-bar delay for weekly close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation - 24-period average volume (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + above 1d EMA34 + volume confirmation
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + below 1d EMA34 + volume confirmation
            elif (close[i] < s4_1w_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below weekly pivot OR below 1d EMA34
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above weekly pivot OR above 1d EMA34
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals