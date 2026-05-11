#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Uses 1-day Camarilla pivot levels (R1/S1) with 1-day trend filter and volume confirmation.
# Long when price breaks above R1 with uptrend (price > EMA34) and volume surge.
# Short when price breaks below S1 with downtrend (price < EMA34) and volume surge.
# Exits on trend reversal or price retracing to EMA34.
# Works in bull markets by riding uptrends, in bear markets by riding downtrends.
# Volume filter reduces false breakouts; EMA34 provides dynamic support/resistance.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla pivot levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.0 / 8.0)
    s1 = pivot - (range_1d * 1.0 / 8.0)
    # Align to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume confirmation (volume > 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 (34) and Camarilla calculation
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1d EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: price breaks above R1 with uptrend and volume surge
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: price breaks below S1 with downtrend and volume surge
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: trend turns down OR price retrace to EMA34
                if downtrend or close[i] < ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price retrace to EMA34
                if uptrend or close[i] > ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals