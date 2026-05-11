#!/usr/bin/env python3
# 1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume
# Hypothesis: Uses 4h Camarilla pivot levels (R1/S1) for breakout entries, filtered by 1d trend (EMA50) and volume surge.
# Long when 1d EMA50 rising, price breaks above 4h R1 with volume confirmation; short when 1d EMA50 falling, price breaks below 4h S1 with volume.
# 1h timeframe for precise entry timing, 4h for Camarilla levels, 1d for trend filter.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by following the 1d trend.
# Volume filter avoids false breakouts; Camarilla levels provide objective support/resistance.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Extract price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA50 for trend direction ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_1d_slope[0] = 0
    ema_50_1d_slope = pd.Series(ema_50_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_slope)
    
    # --- 4h Camarilla pivot levels (R1, S1) ---
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Typical price for pivot calculation
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    # Camarilla levels
    camarilla_width = (high_4h - low_4h) * 1.1 / 12.0
    r1_4h = close_4h + camarilla_width * 1.0  # R1 = C + (H-L)*1.1/12
    s1_4h = close_4h - camarilla_width * 1.0  # S1 = C - (H-L)*1.1/12
    
    # Align Camarilla levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # --- Volume confirmation (volume > 24-period average on 1h) ---
    vol_ma = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1d EMA50 (50) and volume MA (24)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_slope_aligned[i]) or
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1d EMA50 slope
        uptrend = ema_50_1d_slope_aligned[i] > 0
        downtrend = ema_50_1d_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 1d uptrend + volume surge + price breaks above 4h R1
                if close[i] > r1_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 1d downtrend + volume surge + price breaks below 4h S1
                if close[i] < s1_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        else:
            if position == 1:
                # Exit long: 1d trend turns down OR price breaks below 4h S1
                if downtrend or close[i] < s1_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: 1d trend turns up OR price breaks above 4h R1
                if uptrend or close[i] > r1_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals