# 6h_WeeklyTrend_DailyRangeBreakout_Volume
# Hypothesis: Combines weekly trend filter (EMA20) with daily range breakout (ATR-based) and volume confirmation.
# Weekly trend ensures alignment with higher timeframe direction. Daily range breakout captures volatility expansion.
# Volume confirms breakout strength. Designed for low trade frequency (15-30/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear via volatility expansion plays.

name = "6h_WeeklyTrend_DailyRangeBreakout_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend (more stable than SMA)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily timeframe
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Calculate daily open (using previous day's close as reference for breakout)
    # For simplicity, use close as proxy for open in daily context
    daily_open = close_1d  # This is acceptable for breakout calculations
    
    # Breakout levels: daily open ± ATR(14) * multiplier
    multiplier = 1.5
    upper_break = daily_open + (atr_14 * multiplier)
    lower_break = daily_open - (atr_14 * multiplier)
    
    # Align daily levels to 6h timeframe
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    
    # Volume confirmation (20-period average on 6h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14) + 5  # need enough history
    
    for i in range(start_idx, n):
        if np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirm = volume[i] > 1.8 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above upper level with volume, above weekly EMA20 (uptrend)
            if close[i] > upper_break_aligned[i] and volume_confirm and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower level with volume, below weekly EMA20 (downtrend)
            elif close[i] < lower_break_aligned[i] and volume_confirm and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower break level or below weekly EMA20
            if close[i] < lower_break_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper break level or above weekly EMA20
            if close[i] > upper_break_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals