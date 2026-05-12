#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from 12h data provide institutional support/resistance.
# Long when price breaks above R1 with volume confirmation and 12h uptrend.
# Short when price breaks below S1 with volume confirmation and 12h downtrend.
# Exit when price returns to the 12h VWAP (mean reversion to fair value).
# Works in trending markets via 12h trend filter and avoids false breakouts with volume confirmation.
# Timeframe: 4h, HTF: 12h for trend and pivot calculation.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # PP = (high + low + close)/3  # Not used directly but needed for context
    range_12h = high_12h - low_12h
    r1_level = close_12h + 1.1 * range_12h / 12
    s1_level = close_12h - 1.1 * range_12h / 12
    
    # Align pivot levels to 4h timeframe (they are constant until next 12h bar)
    r1_4h = align_htf_to_ltf(prices, df_12h, r1_level)
    s1_4h = align_htf_to_ltf(prices, df_12h, s1_level)
    
    # 12h trend filter: EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h VWAP for exit (mean reversion target)
    # VWAP = sum(price * volume) / sum(volume) over session
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    vwap_12h = (typical_price_12h * volume_12h).cumsum() / volume_12h.cumsum()
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vwap_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get current values
        r1 = r1_4h[i]
        s1 = s1_4h[i]
        ema50 = ema50_12h_aligned[i]
        vwap = vwap_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        trend_up = close[i] > ema50
        trend_down = close[i] < ema50
        
        if position == 0:
            # LONG: price breaks above R1 with volume confirmation and 12h uptrend
            if close[i] > r1 and vol_conf and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with volume confirmation and 12h downtrend
            elif close[i] < s1 and vol_conf and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price returns to VWAP (mean reversion) or trend fails
            if close[i] <= vwap or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to VWAP (mean reversion) or trend fails
            if close[i] >= vwap or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals