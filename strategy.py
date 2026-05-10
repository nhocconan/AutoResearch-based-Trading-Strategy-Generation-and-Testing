#!/usr/bin/env python3
# 1d_WideRangeBreakout_1wTrend_Volume
# Hypothesis: On daily timeframe, enter long when price breaks above 20-day high with volume > 1.5x average and weekly trend up (price > weekly EMA50). Enter short when price breaks below 20-day low with volume > 1.5x average and weekly trend down (price < weekly EMA50). Uses ATR-based stoploss. Designed for 10-25 trades/year to avoid fee drag. Weekly trend filter ensures alignment with higher timeframe momentum, working in both bull and bear markets by only trading in the direction of the weekly trend.

name = "1d_WideRangeBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 20-day high/low for breakout
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.nanmax(high[i-20:i])
        low_20[i] = np.nanmin(low[i-20:i])
    
    # 20-day volume average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of weekly EMA50 trend
            if close[i] > ema_50_1w_aligned[i]:  # Weekly uptrend
                # Long: Breakout above 20-day high with volume confirmation
                if close[i] > high_20[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Weekly downtrend
                # Short: Breakout below 20-day low with volume confirmation
                if close[i] < low_20[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below weekly EMA50 or stoploss hit (2*ATR below entry)
            if close[i] < ema_50_1w_aligned[i] or (i > 0 and low[i] < high_20[i] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above weekly EMA50 or stoploss hit (2*ATR above entry)
            if close[i] > ema_50_1w_aligned[i] or (i > 0 and high[i] > low_20[i] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals