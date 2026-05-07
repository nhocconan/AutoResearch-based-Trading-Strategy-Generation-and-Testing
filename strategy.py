# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1h_Engulfing_4hTrend_1dVolumeFilter
Hypothesis: Use 4h EMA for trend direction, 1d volume filter to avoid low-liquidity noise, and 1h bullish/bearish engulfing patterns for precise entries. Works in bull/bear by following higher timeframe trend with volume confirmation. Target: 20-40 trades/year.
"""

name = "1h_Engulfing_4hTrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mats_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1h data
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 1h engulfing detection
    bullish_engulf = (close > open_) & (open_ > close) & (close > open_.shift(1)) & (open_ < close.shift(1))
    bearish_engulf = (close < open_) & (open_ < close) & (close < open_.shift(1)) & (open_ > close.shift(1))
    # Fix first element
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Warmup for 4h EMA and 1d volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend determination from 4h EMA
        trend_up = close[i] > ema_21_4h_aligned[i]
        trend_down = close[i] < ema_21_4h_aligned[i]
        
        # Volume filter: current 1h volume > 20-day average 1d volume
        vol_filter = volume[i] > vol_ma20_1d_aligned[i]
        
        if position == 0:
            # Long: bullish engulfing in uptrend with volume filter
            if bullish_engulf[i] and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: bearish engulfing in downtrend with volume filter
            elif bearish_engulf[i] and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: bearish engulfing or trend reversal
            if bearish_engulf[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: bullish engulfing or trend reversal
            if bullish_engulf[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals