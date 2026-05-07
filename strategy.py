#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12-hour Camarilla R1/S1 breakout with daily trend filter and volume confirmation.
# Camarilla levels provide institutional support/resistance. Works in bull markets via breakouts
# at R1 (resistance) and in bear markets via breakdowns at S1 (support). Volume filter reduces
# false signals, daily trend filter avoids counter-trend trades. Target: 12-37 trades per year
# (~50-150 over 4 years) with position size 0.25.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from previous 12h bar
    # Formula: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), 
    #          R2 = close + 0.75*(high-low), R1 = close + 0.375*(high-low)
    #          S1 = close - 0.375*(high-low), S2 = close - 0.75*(high-low),
    #          S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use R1 and S1 as entry levels
    range_12h = np.zeros(n)
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    
    # Calculate for each bar using previous bar's high/low/close
    for i in range(1, n):
        if i-1 >= 0:
            h = high[i-1]
            l = low[i-1]
            c = close[i-1]
            range_val = h - l
            range_12h[i] = range_val
            r1[i] = c + 0.375 * range_val
            s1[i] = c - 0.375 * range_val
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1[i]   # Break above R1
        breakout_down = close[i] < s1[i] # Break below S1
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: upward breakout at R1 + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout at S1 + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below R1 or trend reversal
            if close[i] < r1[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above S1 or trend reversal
            if close[i] > s1[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals