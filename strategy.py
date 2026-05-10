#!/usr/bin/env python3
# 6h_Keltner_Channel_Breakout_1dTrend_Volume
# Hypothesis: Keltner Channel breakout on 6h with 1d EMA trend filter and volume confirmation.
# Uses volatility-based bands (ATR) to capture breakouts in both trending and ranging markets.
# Trend filter ensures we only trade in direction of higher timeframe momentum.
# Volume confirmation filters out false breakouts.
# Target: 20-50 trades per year on 6h timeframe.

name = "6h_Keltner_Channel_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def average_true_range(high, low, close, period):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    atr = pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean()
    return atr.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for Keltner Channels on 6h
    atr_20 = average_true_range(high, low, close, 20)
    
    # Calculate EMA(20) for middle band
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    upper_band = ema_20 + (2 * atr_20)
    lower_band = ema_20 - (2 * atr_20)
    
    # Volume filter: current volume > 1.5x 20-period EMA of volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) + EMA20 (20) + ATR (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above upper band AND price above 1d EMA50 AND volume filter
            if close[i] > upper_band[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower band AND price below 1d EMA50 AND volume filter
            elif close[i] < lower_band[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below middle band (EMA20) OR trend reversal
            if close[i] < ema_20[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above middle band (EMA20) OR trend reversal
            if close[i] > ema_20[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals