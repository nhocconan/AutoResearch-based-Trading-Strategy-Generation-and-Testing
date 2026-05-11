#!/usr/bin/env python3
# 4h_Donchian20_Trend_Follow_1dEMA34
# Hypothesis: Combines 4h Donchian breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, 1d EMA34 rising, and volume surge.
# Short when price breaks below Donchian(20) low, 1d EMA34 falling, and volume surge.
# Exits when price crosses back below/above Donchian(20) midpoint or trend reverses.
# Designed for 4h timeframe to capture trends with controlled trade frequency (~20-50/year).
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by following higher timeframe trend.

name = "4h_Donchian20_Trend_Follow_1dEMA34"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mpf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend direction ---
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_1d_slope[0] = 0
    ema_34_1d_slope = pd.Series(ema_34_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_slope)
    
    # --- 4h Donchian(20) channels ---
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian (20) and EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(ema_34_1d_slope_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1d EMA34 slope
        uptrend = ema_34_1d_slope_aligned[i] > 0
        downtrend = ema_34_1d_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 1d uptrend + volume surge + break above Donchian high
                if close[i] > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 1d downtrend + volume surge + break below Donchian low
                if close[i] < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 1d trend turns down OR price crosses below Donchian mid
                if downtrend or close[i] < donchian_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 1d trend turns up OR price crosses above Donchian mid
                if uptrend or close[i] > donchian_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals