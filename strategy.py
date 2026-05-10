#!/usr/bin/env python3
"""
4h_Bollinger_Band_Width_Trend_1d
Hypothesis: Bollinger Band Width (BBW) measures volatility regime; low BBW indicates squeeze (potential breakout), high BBW indicates expansion (trend). Combine with 1d trend filter and volume confirmation to trade breakouts from squeezes in the direction of the higher timeframe trend. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend). Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "4h_Bollinger_Band_Width_Trend_1d"
timeframe = "4h"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    if n >= bb_period:
        # Calculate SMA
        for i in range(bb_period - 1, n):
            sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        # Calculate standard deviation
        for i in range(bb_period - 1, n):
            std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
        upper = sma + bb_std * std_dev
        lower = sma - bb_std * std_dev
        bb_width = (upper - lower) / sma  # Normalized width
    else:
        bb_width = np.full(n, np.nan)
    
    # Bollinger Band Width percentile rank (50-period lookback) to identify squeeze
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    if n >= lookback:
        for i in range(lookback - 1, n):
            # Calculate percentile rank of current BBW over lookback window
            window = bb_width[i - lookback + 1:i + 1]
            # Remove NaNs
            window = window[~np.isnan(window)]
            if len(window) > 0:
                # Percentile rank: percentage of values in window less than current value
                bb_width_percentile[i] = (np.sum(window < bb_width[i]) / len(window)) * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, bb_period, lookback)
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(bb_width_percentile[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        # Squeeze condition: BBW percentile < 20 (low volatility)
        squeeze = bb_width_percentile[i] < 20
        
        if position == 0:
            # Long: BBW squeeze breakout above upper band in uptrend with volume confirmation
            if squeeze and bb_width_percentile[i] >= 20 and close[i] > upper[i] and close[i] > ema50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: BBW squeeze breakdown below lower band in downtrend with volume confirmation
            elif squeeze and bb_width_percentile[i] >= 20 and close[i] < lower[i] and close[i] < ema50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: BBW expansion (percentile > 80) or trend reversal
            if bb_width_percentile[i] > 80 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: BBW expansion (percentile > 80) or trend reversal
            if bb_width_percentile[i] > 80 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals