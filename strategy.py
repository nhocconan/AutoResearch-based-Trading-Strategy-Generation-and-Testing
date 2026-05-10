#!/usr/bin/env python3
# 4H_BollingerBand_Breakout_Volume_Trend_Filter
# Hypothesis: Bollinger Band breakouts above the upper band indicate strong upward momentum, while breaks below the lower band indicate strong downward momentum.
# Combined with volume confirmation (volume > 2x 20-period average) to filter false breakouts and a 1-day EMA50 trend filter to ensure alignment with the higher timeframe trend.
# Designed for low trade frequency (~15-30/year) with discrete sizing (0.25) to minimize fee drag. Works in both bull and bear markets by following the trend.

name = "4H_BollingerBand_Breakout_Volume_Trend_Filter"
timeframe = "4h"
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
    
    # Bollinger Bands: 20-period SMA with 2 standard deviations
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    # Daily trend filter: EMA 50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get daily close for trend determination
        close_1d_series = pd.Series(close_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_series.values)
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above upper Bollinger Band + volume confirmation + daily uptrend
            if close[i] > upper_band[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower Bollinger Band + volume confirmation + daily downtrend
            elif close[i] < lower_band[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below the 20-period SMA (mean reversion signal)
            if close[i] < sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above the 20-period SMA (mean reversion signal)
            if close[i] > sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals