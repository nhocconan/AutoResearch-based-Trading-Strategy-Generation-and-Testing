#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSurge"
timeframe = "12h"
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
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # Volume surge: current volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i - 20]
        if i < 19:
            vol_ma20[i] = 0.0
        else:
            vol_ma20[i] = vol_sum / 20.0
    volume_surge = volume > (vol_ma20 * 1.5)
    
    # Camarilla levels from previous 1d bar
    R1 = np.zeros(n)
    S1 = np.zeros(n)
    for i in range(n):
        if i == 0:
            R1[i] = 0.0
            S1[i] = 0.0
        else:
            # Previous 1d bar's high, low, close
            # Since we're on 12h timeframe, we need to get the previous day's values
            # We'll use the high/low/close from 2 bars ago (since 12h * 2 = 24h)
            if i >= 2:
                R1[i] = close[i-1] + (high[i-2] - low[i-2]) * 1.1 / 12
                S1[i] = close[i-1] - (high[i-2] - low[i-2]) * 1.1 / 12
            else:
                R1[i] = 0.0
                S1[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~24 hours
    
    start_idx = 2  # Need at least 2 bars for Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i]) or 
            np.isnan(R1[i]) or 
            np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above R1 AND 1d uptrend AND volume surge
            if close[i] > R1[i] and trend_up[i] and volume_surge[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S1 AND 1d downtrend AND volume surge
            elif close[i] < S1[i] and trend_down[i] and volume_surge[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below S1 OR trend turns down OR volume drops
            if close[i] < S1[i] or not trend_up[i] or not volume_surge[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R1 OR trend turns up OR volume drops
            if close[i] > R1[i] or not trend_down[i] or not volume_surge[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 1d trend filter and volume surge confirmation.
# Long when price breaks above R1 in 1d uptrend with volume surge, short when breaks below S1 in 1d downtrend with volume surge.
# Uses 12h timeframe for entries, 1d for trend filter. Volume surge ensures breakouts are genuine.
# Cooldown of 2 bars (~24h) limits trades. Position size 0.25 manages risk.
# Works in bull markets (captures uptrend continuations) and bear markets (captures downtrends).