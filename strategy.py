#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
In trending markets (price > 1d EMA34), buy R3 breakouts or sell S3 breakdowns with volume > 1.5x average.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year.
Works in bull/bear via trend alignment: only trade in direction of 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 20-period volume average for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg_20)
    
    # Calculate Camarilla levels from previous 12h bar
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using typical Camarilla formula: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # Simplified: range = high - low, R3 = close + range * 1.1/4, S3 = close - range * 1.1/4
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_12h = prev_high - prev_low
    r3 = prev_close + (range_12h * 1.1 / 4)
    s3 = prev_close - (range_12h * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume avg, 1 for Camarilla)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(r3[i]) or np.isnan(s3[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        vol_spike = volume_spike[i]
        
        # Trend alignment condition
        trend_up = htf_trend[i] == 1
        trend_down = htf_trend[i] == -1
        
        # Breakout conditions
        breakout_up = close[i] > r3[i]
        breakout_down = close[i] < s3[i]
        
        # Exit conditions: return to midpoint or opposite Camarilla level
        midpoint = (r3[i] + s3[i]) / 2
        exit_long = close[i] < midpoint
        exit_short = close[i] > midpoint
        
        if trend_up and vol_spike and breakout_up:
            # Long breakout in uptrend with volume spike
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif trend_down and vol_spike and breakout_down:
            # Short breakdown in downtrend with volume spike
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        elif position == 1 and exit_long:
            # Exit long position
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            # Exit short position
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0