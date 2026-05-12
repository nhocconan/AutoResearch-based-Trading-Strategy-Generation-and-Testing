#!/usr/bin/env python3
"""
6H_WEEKLY_PIVOT_REVERSION
Hypothesis: Trade reversals at weekly pivot levels (R4/S4) with 1-day volume confirmation and 1-day trend filter.
Works in bull/bear markets by fading extremes when institutional levels are tested with volume,
and only taking trades in direction of higher timeframe trend to avoid counter-trend whipsaws.
Target: 15-30 trades/year on 6h.
"""
name = "6H_WEEKLY_PIVOT_REVERSION"
timeframe = "6h"
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
    
    # Calculate weekly pivot points from previous week
    # Use Monday's open as week start approximation (simplified)
    # For true weekly pivot: need weekly OHLC, approximate using rolling
    weekly_high = pd.Series(high).rolling(window=20, min_periods=20).max().values  # ~5 days * 4 = 20 (6h bars)
    weekly_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    weekly_close = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Pivot = (H+L+C)/3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    # R4 = R3 + (R3 - S3) where R3 = P + 2*(H-L), S3 = P - 2*(H-L)
    # So R4 = P + 3*(H-L), S4 = P - 3*(H-L)
    weekly_range = weekly_high - weekly_low
    weekly_r4 = pivot + 3 * weekly_range
    weekly_s4 = pivot - 3 * weekly_range
    
    # 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1-day volume spike confirmation (current vs 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d / vol_ma_20d
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for weekly calculations
        if (np.isnan(weekly_r4[i]) or np.isnan(weekly_s4[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches S4 with volume spike and above daily EMA34 (bullish bias)
            if (low[i] <= weekly_s4[i] and 
                vol_spike_aligned[i] > 1.8 and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R4 with volume spike and below daily EMA34 (bearish bias)
            elif (high[i] >= weekly_r4[i] and 
                  vol_spike_aligned[i] > 1.8 and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above weekly pivot (mean reversion complete)
            if close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below weekly pivot (mean reversion complete)
            if close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals