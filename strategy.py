#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels (R1/S1) from daily high/low/close to identify
# intraday support/resistance. Enter long when price breaks above R1 with volume confirmation
# and price above daily EMA34 (bullish trend). Enter short when price breaks below S1 with
# volume confirmation and price below daily EMA34 (bearish trend). Exit on reversion to the
# daily pivot point (PP) or trend failure. Designed for 4h timeframe to capture multi-day
# swings while avoiding noise. Works in bull markets (catching breakouts) and bear markets
# (catching breakdowns) with trend filter and volume confirmation to reduce false signals.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H - L) * 1.12
    # S1 = C - (H - L) * 1.12
    # PP = (H + L + C) / 3
    diff = high_1d - low_1d
    r1 = close_1d + diff * 1.12
    s1 = close_1d - diff * 1.12
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    pp_4h = align_htf_to_ltf(prices, df_1d, pp)
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(pp_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_4h[i]
        trend_down = close[i] < ema_34_1d_4h[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Price levels
        r1_level = r1_4h[i]
        s1_level = s1_4h[i]
        pp_level = pp_4h[i]
        
        if position == 0:
            # LONG: Break above R1 with volume confirmation and bullish trend
            if close[i] > r1_level and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume confirmation and bearish trend
            elif close[i] < s1_level and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Return to pivot point or trend failure
            if close[i] <= pp_level or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Return to pivot point or trend failure
            if close[i] >= pp_level or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals