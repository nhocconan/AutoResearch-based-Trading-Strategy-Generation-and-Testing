#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Breakout_1dTrend_Volume
# Hypothesis: Go long when price breaks above Camarilla R4 with volume > 1.5x average in uptrend (price > 1d EMA34).
# Go short when price breaks below Camarilla S4 with volume > 1.5x average in downtrend (price < 1d EMA34).
# Exit when price crosses back through the Camarilla pivot point (PP) or ATR-based stoploss hit.
# Uses daily Camarilla levels for institutional support/resistance, works in both bull and bear markets by following 1d trend.
# Designed for 12-37 trades/year to avoid fee drag on 12h timeframe.

name = "12h_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        ema_34_1d[i] = np.mean(close_1d[i-34:i])
    
    # Align daily EMA34 to 12h timeframe (waits for daily close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1 = pp + (range_1d * 1.0833)
    r2 = pp + (range_1d * 1.1666)
    r3 = pp + (range_1d * 1.2500)
    r4 = pp + (range_1d * 1.5000)
    
    # Support levels
    s1 = pp - (range_1d * 1.0833)
    s2 = pp - (range_1d * 1.1666)
    s3 = pp - (range_1d * 1.2500)
    s4 = pp - (range_1d * 1.5000)
    
    # Align all Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of daily EMA34 trend
            if close[i] > ema_34_1d_aligned[i]:  # Uptrend
                # Long: Break above R4 with volume confirmation
                if close[i] > r4_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Break below S4 with volume confirmation
                if close[i] < s4_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price crosses back through pivot point or stoploss hit
            if close[i] < pp_aligned[i] or (i > 0 and low[i] < ema_34_1d_aligned[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses back through pivot point or stoploss hit
            if close[i] > pp_aligned[i] or (i > 0 and high[i] > ema_34_1d_aligned[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals