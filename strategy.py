#!/usr/bin/env python3
# 4h_12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
# Hypothesis: Breakout from daily Camarilla R3/S3 levels in direction of 12h trend, confirmed by volume spike.
# Works in bull markets (riding uptrends above R3) and bear markets (riding downtrends below S3).
# Uses volume to avoid false breakouts and 12h EMA34 for trend filter.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (using previous day's OHLC) ---
    # Calculate pivot points from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla equations
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h timeframe (values valid after previous day close)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # --- 12h EMA34 for trend direction ---
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_slope = ema_34_12h - np.roll(ema_34_12h, 1)
    ema_34_12h_slope[0] = 0
    ema_34_12h_slope = pd.Series(ema_34_12h_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_34_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_slope)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 (34) and Camarilla (need prev day)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or
            np.isnan(ema_34_12h_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 12h EMA34 slope
        uptrend = ema_34_12h_slope_aligned[i] > 0
        downtrend = ema_34_12h_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 12h uptrend + volume surge + price breaks above R3
                if close[i] > R3_4h[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 12h downtrend + volume surge + price breaks below S3
                if close[i] < S3_4h[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 12h trend turns down OR price crosses below S3 (reversal level)
                if downtrend or close[i] < S3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 12h trend turns up OR price crosses above R3 (reversal level)
                if uptrend or close[i] > R3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals