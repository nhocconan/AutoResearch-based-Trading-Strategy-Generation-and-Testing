#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 AND 12h close > 12h EMA34 AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND 12h close < 12h EMA34 AND volume > 1.5x 20-period average
# Exit when Elder Power crosses zero (momentum shift) OR price touches 12h EMA34 (mean reversion)
# Uses 6h primary timeframe with 12h HTF for all indicators (Elder Ray, EMA34)
# Elder Ray measures bull/bear power relative to EMA13, filtering weak momentum
# Volume confirmation ensures breakouts have conviction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ElderRay_12hEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for all indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA13 for Elder Ray (Elder Ray uses EMA13)
    ema_13_12h = pd.Series(df_12h['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_12h['high'].values - ema_13_12h
    bear_power = df_12h['low'].values - ema_13_12h
    
    # Align all indicators to 6h timeframe
    ema_13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND 12h close > 12h EMA34 AND volume spike
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND 12h close < 12h EMA34 AND volume spike
            elif (bear_power_aligned[i] < 0 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses below zero (momentum loss) OR price touches 12h EMA34 (mean reversion)
            if bull_power_aligned[i] <= 0 or abs(close[i] - ema_34_12h_aligned[i]) < 0.001 * ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses above zero (momentum loss) OR price touches 12h EMA34 (mean reversion)
            if bear_power_aligned[i] >= 0 or abs(close[i] - ema_34_12h_aligned[i]) < 0.001 * ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals