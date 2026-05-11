#!/usr/bin/env python3
# 4h_Keltner_Breakout_12hTrend
# Hypothesis: Captures breakouts with momentum by combining 12h trend direction (EMA200)
# with Keltner Channel breakouts on 4h, confirmed by volume surge. Works in both bull and bear markets
# by following the higher timeframe trend. Uses ATR-based stop via position reversal when trend changes.
# Designed for low trade frequency (<400 total 4h trades) to minimize fee drag.

name = "4h_Keltner_Breakout_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 12h data for EMA200 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA200 for trend direction ---
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_slope = ema_200_12h - np.roll(ema_200_12h, 1)
    ema_200_12h_slope[0] = 0
    ema_200_12h_slope = pd.Series(ema_200_12h_slope).ewm(span=5, adjust=False, min_periods=1).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    ema_200_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h_slope)
    
    # --- 4h Keltner Channel (20, 1.5) ---
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_20 + 1.5 * atr
    keltner_lower = ema_20 - 1.5 * atr
    
    # --- Volume confirmation (volume > 30-period average) ---
    vol_ma = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA200 (200) and EMA20/ATR (20)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_12h_aligned[i]) or
            np.isnan(ema_200_12h_slope_aligned[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(atr[i]) or
            np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 12h EMA200 slope
        uptrend = ema_200_12h_slope_aligned[i] > 0
        downtrend = ema_200_12h_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 12h uptrend + volume surge + close above Keltner upper
                if close[i] > keltner_upper[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 12h downtrend + volume surge + close below Keltner lower
                if close[i] < keltner_lower[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 12h trend turns down OR close below Keltner lower
                if downtrend or close[i] < keltner_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 12h trend turns up OR close above Keltner upper
                if uptrend or close[i] > keltner_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals