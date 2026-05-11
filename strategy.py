#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Breakout above Camarilla R3 or below S3 with 1d EMA trend filter and volume confirmation. Works in bull/bear markets by trading breakouts in trend direction. Uses 12h timeframe for low trade frequency to minimize fee drag. Targets 15-30 trades/year.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # === 1d OHLC for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d EMA34 Trend Filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter (12h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # 2x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 + above 1d EMA34 + volume spike
            if (close[i] > r3_12h[i] and 
                close[i] > ema34_12h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Close breaks below S3 + below 1d EMA34 + volume spike
            elif (close[i] < s3_12h[i] and 
                  close[i] < ema34_12h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (4 bars)
            holding_bars += 1
            if holding_bars < 4:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Close returns to 1d EMA34
            if position == 1:
                if close[i] < ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals