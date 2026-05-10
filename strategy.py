#!/usr/bin/env python3
# 4h_1d_Keltner_Breakout_Trend_Filter
# Hypothesis: Breakouts from daily Keltner Channels with 1d trend and volume confirmation.
# Daily Keltner Channels (EMA-based ATR bands) provide adaptive support/resistance.
# Breakouts above upper band in uptrend or below lower band in downtrend capture momentum.
# Volume surge confirms breakout validity. Works in bull/bear via directional filtering.
# Targets 20-40 trades/year to minimize fee drag.

name = "4h_1d_Keltner_Breakout_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Keltner Channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Keltner Channels (EMA-based ATR bands)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA20 for middle line
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR10 for band width
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index 0
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channels: Upper = EMA20 + 2*ATR10, Lower = EMA20 - 2*ATR10
    upper_1d = ema_20_1d + 2.0 * atr_10_1d
    lower_1d = ema_20_1d - 2.0 * atr_10_1d
    
    # Align daily Keltner levels to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Get 1d data for trend filter (using EMA50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Keltner (20) + volume MA (20) + EMA (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Price position relative to daily Keltner Channels
        price_above_upper = close[i] > upper_1d_aligned[i]
        price_below_lower = close[i] < lower_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner band with volume surge and 1d uptrend
            if price_above_upper and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band with volume surge and 1d downtrend
            elif price_below_lower and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below upper band OR trend changes
            if close[i] < upper_1d_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above lower band OR trend changes
            if close[i] > lower_1d_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals