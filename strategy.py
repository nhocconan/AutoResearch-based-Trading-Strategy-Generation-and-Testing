#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Use Camarilla pivot levels from daily to identify key support/resistance.
Go long when price breaks above R1 with 1d EMA34 uptrend and volume spike.
Go short when price breaks below S1 with 1d EMA34 downtrend and volume spike.
Camarilla levels provide statistically significant intraday support/resistance.
1d EMA34 filters for trend direction to avoid counter-trend trades.
Volume confirmation ensures breakouts have participation.
Designed for low trade frequency (~12-37/year) with high win rate by requiring confluence.
Works in both bull and bear markets by following the 1d trend direction.
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate R1 and S1 for each day
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1d data for EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2x 20-period average volume (strict for fewer trades)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema_34_12h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price breakout conditions
        breakout_above_r1 = close[i] > r1_12h[i]
        breakdown_below_s1 = close[i] < s1_12h[i]
        
        trend_up = close[i] > ema_34_12h[i]
        trend_down = close[i] < ema_34_12h[i]
        
        if position == 0:
            # Long: break above R1 + 1d uptrend + volume spike
            if breakout_above_r1 and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + 1d downtrend + volume spike
            elif breakdown_below_s1 and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or trend reversal
            if close[i] < s1_12h[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or trend reversal
            if close[i] > r1_12h[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals