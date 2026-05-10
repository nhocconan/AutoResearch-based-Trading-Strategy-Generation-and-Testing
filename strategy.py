#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Uses Camarilla pivot levels from 1d timeframe for entry signals, with 12h trend filter and volume confirmation.
# In bear markets: Fade at R3/S3 levels (mean reversion). In bull markets: Breakout continuation at R4/S4 levels (trend following).
# Uses 12h EMA50 for trend direction filter. Volume > 1.5x 20-period average for confirmation.
# Designed to work in both bull (breakout continuation) and bear (mean reversion at extremes) markets.
# Target: 15-30 trades/year (~60-120 total over 4 years) to stay within optimal trade frequency for 6h.

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    volume = prices['volume'].values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    R3 = prev_close + range_1d * 1.1 / 4
    S3 = prev_close - range_1d * 1.1 / 4
    R4 = prev_close + range_1d * 1.1 / 2
    S4 = prev_close - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bull market (price > EMA50): Breakout continuation at R4/S4
            if close[i] > ema_50_aligned[i]:
                # Long breakout above R4
                if close[i] > R4_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S4
                elif close[i] < S4_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            # Bear market (price <= EMA50): Mean reversion at R3/S3
            else:
                # Long mean reversion from S3
                if close[i] < S3_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short mean reversion from R3
                elif close[i] > R3_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            if close[i] > ema_50_aligned[i]:
                # In bull trend: exit on breakdown below S3 (mean reversion level)
                if close[i] < S3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In bear trend: exit on reversion to mean (middle)
                if close[i] > (R3_aligned[i] + S3_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            if close[i] > ema_50_aligned[i]:
                # In bull trend: exit on reversion to mean (middle)
                if close[i] < (R3_aligned[i] + S3_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In bear trend: exit on breakout above R3 (continuation)
                if close[i] > R3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals