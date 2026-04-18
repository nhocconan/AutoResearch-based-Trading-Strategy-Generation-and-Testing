#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA34 filter and volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA13 to detect trend strength.
# Uses 13-period EMA for Elder Ray calculation and 34-period EMA on 12h timeframe as trend filter.
# Volume confirmation ensures institutional participation. Designed for low trade frequency (15-35/year).
# Works in bull markets (strong bull power above zero) and bear markets (strong bear power below zero).
name = "6h_ElderRay_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (using close prices)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 12h data for EMA34 filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 34-period EMA on 12h close
    close_12h = pd.Series(df_12h['close'].values)
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 6h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume above 1.5x average
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price above 12h EMA34 AND volume spike
            long_condition = (bull_power[i] > 0) and (close[i] > ema34_12h_aligned[i]) and volume_spike
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price below 12h EMA34 AND volume spike
            elif (bear_power[i] < 0) and (close[i] < ema34_12h_aligned[i]) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative OR price crosses below 12h EMA34
            exit_condition = (bull_power[i] <= 0) or (close[i] < ema34_12h_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive OR price crosses above 12h EMA34
            exit_condition = (bear_power[i] >= 0) or (close[i] > ema34_12h_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals