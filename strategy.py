#!/usr/bin/env python3
# 12h_1w_Trend_With_Volume_Confirmation
# Hypothesis: Uses 1-week EMA40 trend direction with 12h price action and volume confirmation.
# Long when 1w EMA40 rising and 12h close above 12h EMA25 + volume surge; short when 1w EMA40 falling and 12h close below 12h EMA25 + volume surge.
# Exits when trend reverses or price crosses EMA25. Designed for fewer trades (<50/year) to avoid fee drag in ranging markets like 2025.
# Works in bull markets by riding uptrends and bear markets by riding downtrends via 1-week trend filter.

name = "12h_1w_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1w data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w EMA40 for trend direction ---
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_slope = ema_40_1w - np.roll(ema_40_1w, 1)
    ema_40_1w_slope[0] = 0
    ema_40_1w_slope = pd.Series(ema_40_1w_slope).ewm(span=3, adjust=False, min_periods=1).mean().values  # smooth slope
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    ema_40_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w_slope)
    
    # --- 12h EMA25 for entry timing ---
    ema_25_12h = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # --- Volume confirmation (volume > 24-period average) ---
    vol_ma = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA25 (25) and EMA40 slope (40+3)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_40_1w_aligned[i]) or
            np.isnan(ema_40_1w_slope_aligned[i]) or
            np.isnan(ema_25_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1w EMA40 slope
        uptrend = ema_40_1w_slope_aligned[i] > 0
        downtrend = ema_40_1w_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 1w uptrend + volume surge + price above 12h EMA25
                if close[i] > ema_25_12h[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 1w downtrend + volume surge + price below 12h EMA25
                if close[i] < ema_25_12h[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 1w trend turns down OR price crosses below EMA25
                if downtrend or close[i] < ema_25_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 1w trend turns up OR price crosses above EMA25
                if uptrend or close[i] > ema_25_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals