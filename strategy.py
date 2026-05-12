#!/usr/bin/env python3
# 6h_WilliamsVixFix_1dTrend
# Hypothesis: Williams Vix Fix (WVF) identifies market bottoms in bearish conditions and tops in bullish conditions.
# Combined with 1d trend filter (EMA50) and volume confirmation to trade reversals with the trend.
# WVF > 0.8 signals extreme fear/greed. Long when WVF > 0.8 and price above 1d EMA50.
# Short when WVF > 0.8 and price below 1d EMA50. Designed for low frequency (15-30 trades/year)
# to capture reversal opportunities in both bull and bear markets by aligning with higher timeframe trend.

name = "6h_WilliamsVixFix_1dTrend"
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
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams Vix Fix (WVF) on 6h ===
    # WVF = ((Highest Close in period - Low) / Highest Close in period) * 100
    # We use 22-period lookback (similar to VIX calculation)
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    wvf = ((highest_close - low) / highest_close) * 100
    # Normalize to 0-1 range for easier thresholding
    wvf_normalized = wvf / 100.0
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(wvf_normalized[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # WVF condition: extreme fear/greed (above 0.8 threshold)
        wvf_extreme = wvf_normalized[i] > 0.8
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Extreme fear/greed + uptrend + volume
            if wvf_extreme and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Extreme fear/greed + downtrend + volume
            elif wvf_extreme and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: When extreme condition passes or trend reverses
            if not wvf_extreme or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: When extreme condition passes or trend reverses
            if not wvf_extreme or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals