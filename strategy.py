#!/usr/bin/env python3
# 6H_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Trade breakouts of Camarilla R3/S3 levels in direction of 12h EMA50 trend
# with volume confirmation. Works in bull/bear by following trend and using volume to
# filter false breakouts. Target: 15-30 trades/year per symbol.

name = "6H_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need previous day's close, high, low
    # Use 1d data to get daily OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3 for previous day
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 6h timeframe (previous day's levels are available at next bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: price above EMA50 = uptrend, below = downtrend
    trend_12h = close_12h > ema50_12h  # True for uptrend
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Trend direction
        is_uptrend = trend_12h_aligned[i] > 0.5
        is_downtrend = trend_12h_aligned[i] < 0.5
        
        if position == 0:
            # Enter long: price breaks above R3 in uptrend with volume
            if is_uptrend and volume_confirm:
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: price breaks below S3 in downtrend with volume
            elif is_downtrend and volume_confirm:
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: trend reverses or price moves back below R3
            if not is_uptrend or close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend reverses or price moves back above S3
            if not is_downtrend or close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals