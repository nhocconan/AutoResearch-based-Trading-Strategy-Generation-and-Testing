#!/usr/bin/env python3
# 6h_1d_RangeExpansion_Breakout_Volume
# Hypothesis: 6h breakout of daily volatility-based range (ATR-based) with daily trend filter and volume confirmation.
# Uses daily ATR to define dynamic range around previous day's close: Range = Close ± 1.5 * ATR(14).
# Long when price breaks above upper range in daily uptrend with volume surge (>1.5x avg), short when breaks below lower range in daily downtrend.
# Designed to capture volatility expansion moves in both bull and bear markets while avoiding chop.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_1d_RangeExpansion_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for ATR-based range and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR with Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(close_1d)
    atr[13] = np.mean(tr[:13])  # Seed with first 14 periods
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pad ATR to match length
    atr = np.concatenate([np.full(14, np.nan), atr])
    
    # Calculate dynamic range: Previous day's close ± 1.5 * ATR
    prev_close = np.roll(close_1d, 1)
    prev_atr = np.roll(atr, 1)
    prev_close[0] = np.nan
    prev_atr[0] = np.nan
    
    range_width = 1.5 * prev_atr
    lower_range = prev_close - range_width
    upper_range = prev_close + range_width
    
    # Align daily ranges to 6h timeframe
    lower_range_aligned = align_htf_to_ltf(prices, df_1d, lower_range)
    upper_range_aligned = align_htf_to_ltf(prices, df_1d, upper_range)
    
    # Daily EMA for trend filter (34-period)
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20-period for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(lower_range_aligned[i]) or
            np.isnan(upper_range_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from daily: close > EMA = uptrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_1d_aligned[i]
        
        # Volume confirmation (1.5x average to balance frequency and reliability)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above upper range in uptrend with volume
            if close[i] > upper_range_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower range in downtrend with volume
            elif close[i] < lower_range_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close back below upper range or trend fails
                if close[i] < upper_range_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close back above lower range or trend fails
                if close[i] > lower_range_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals