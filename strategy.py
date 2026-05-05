#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 12h EMA50 trend filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 measures bull/bear strength
# 12h EMA50 provides major trend direction (avoid counter-trend trades)
# Volume confirmation: current volume > 1.8x 20-period MA to ensure conviction
# Long: Bull Power > 0 AND Bear Power < 0 (bulls in control) AND price > 12h EMA50 AND volume spike
# Short: Bear Power < 0 AND Bull Power < 0 (bears in control) AND price < 12h EMA50 AND volume spike
# Exit: When Elder Ray signals weaken (Bull Power <= 0 for long, Bear Power >= 0 for short) OR price crosses 12h EMA50
# Uses Elder Ray for momentum conviction, 12h EMA50 for trend filter, volume for spam reduction
# Timeframe: 6h, HTF: 12h. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ElderRay_12hEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 6h EMA13 for Elder Ray
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 (bulls in control) 
            # AND price > 12h EMA50 (uptrend) AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power < 0 (bears in control) 
            # AND price < 12h EMA50 (downtrend) AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (bulls losing control) OR price crosses below 12h EMA50
            if bull_power[i] <= 0 or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (bears losing control) OR price crosses above 12h EMA50
            if bear_power[i] >= 0 or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals