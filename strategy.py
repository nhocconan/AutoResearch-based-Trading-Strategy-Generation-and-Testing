#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) + 12h EMA50 Trend Filter + Volume Spike Confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND 12h uptrend AND volume spike
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND 12h downtrend AND volume spike
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing strong trends.
# Works in bull markets via long signals in uptrend and bear markets via short signals in downtrend.
# Uses EMA13 for power calculation (standard) and EMA50 for trend filter (proven effective).

name = "6h_ElderRay_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA13 for Elder Ray power calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power rising (less negative) AND 12h uptrend AND volume spike
            if (bull_power[i] > 0 and 
                i > 100 and bear_power[i] > bear_power[i-1] and  # Bear Power rising
                close[i] > ema_50_aligned[i] and  # 12h uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power falling (less positive) AND 12h downtrend AND volume spike
            elif (bear_power[i] < 0 and 
                  i > 100 and bull_power[i] < bull_power[i-1] and  # Bull Power falling
                  close[i] < ema_50_aligned[i] and  # 12h downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power starts falling OR 12h trend turns down
            if (bull_power[i] <= 0 or 
                i > 100 and bear_power[i] < bear_power[i-1] or  # Bear Power falling
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bull Power starts rising OR 12h trend turns up
            if (bear_power[i] >= 0 or 
                i > 100 and bull_power[i] > bull_power[i-1] or  # Bull Power rising
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals