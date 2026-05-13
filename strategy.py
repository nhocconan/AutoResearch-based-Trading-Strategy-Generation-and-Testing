#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation.
# Camarilla levels calculated from prior 12h bar (HLC). Long when price breaks above R3 with 12h EMA50 uptrend and volume > 1.5x average.
# Short when price breaks below S3 with 12h EMA50 downtrend and volume > 1.5x average.
# Exit on break of opposite Camarilla level (R3/S3) or volume drop below average.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence of breakout, trend, and volume confirmation.
# Camarilla levels identify intraday support/resistance; breakouts with volume and trend alignment capture strong moves.
# Effective in both bull and bear markets by trading breakouts in direction of higher timeframe trend.

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from prior 12h bar: based on (H+L+C)
    # Typical Price = (H+L+C)/3, Camarilla width = (H-L) * 1.1/12
    typical_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    camarilla_width = range_12h * 1.1 / 12
    
    # R3 = typical + 4 * width, S3 = typical - 4 * width
    r3_12h = typical_12h + 4 * camarilla_width
    s3_12h = typical_12h - 4 * camarilla_width
    
    # Align Camarilla levels to 6h timeframe (completed 12h bar only)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3, 12h EMA50 uptrend (close > EMA), volume confirmation
            if close[i] > r3_12h_aligned[i] and close_12h[-1] > ema50_12h[-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3, 12h EMA50 downtrend (close < EMA), volume confirmation
            elif close[i] < s3_12h_aligned[i] and close_12h[-1] < ema50_12h[-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 (opposite level) OR volume drops below average
            if close[i] < s3_12h_aligned[i] or volume[i] < vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R3 (opposite level) OR volume drops below average
            if close[i] > r3_12h_aligned[i] or volume[i] < vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals