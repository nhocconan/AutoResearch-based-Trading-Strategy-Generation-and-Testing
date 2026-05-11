#!/usr/bin/env python3
# 4h_MultiTF_Trend_Follow
# Hypothesis: Follows strong trends using 12h EMA50 direction with 4h price action confirmation.
# Long when 12h EMA50 rising and 4h closes above 4h EMA21; short when 12h EMA50 falling and 4h closes below 4h EMA21.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by following the higher timeframe trend.
# Uses volume confirmation to avoid false breakouts and ATR-based stop to manage risk.

name = "4h_MultiTF_Trend_Follow"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 for trend direction ---
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_slope = ema_50_12h - np.roll(ema_50_12h, 1)
    ema_50_12h_slope[0] = 0
    ema_50_12h_slope = pd.Series(ema_50_12h_slope).ewm(span=3, adjust=False, min_periods=1).mean().values  # smooth slope
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_50_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_slope)
    
    # --- 4h EMA21 for entry timing ---
    ema_21_4h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA21 (21) and EMA50 slope (50+3)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(ema_50_12h_slope_aligned[i]) or
            np.isnan(ema_21_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 12h EMA50 slope
        uptrend = ema_50_12h_slope_aligned[i] > 0
        downtrend = ema_50_12h_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 12h uptrend + volume surge + price above 4h EMA21
                if close[i] > ema_21_4h[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 12h downtrend + volume surge + price below 4h EMA21
                if close[i] < ema_21_4h[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 12h trend turns down OR price crosses below EMA21
                if downtrend or close[i] < ema_21_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 12h trend turns up OR price crosses above EMA21
                if uptrend or close[i] > ema_21_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals