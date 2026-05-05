#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1w trend filter and volume confirmation
# Long when Bull Power > 0 AND Bear Power < 0 AND 1w close > 1w EMA50 AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND Bull Power < 0 AND 1w close < 1w EMA50 AND volume > 1.5x 20-period average
# Exit when Elder Ray signals reverse (Bull Power crosses below 0 for long, Bear Power crosses above 0 for short)
# Uses 6h primary timeframe with 1w HTF for trend filter and Elder Ray for momentum
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Elder Ray measures bull/bear power relative to EMA13; 1w EMA50 filters for higher-timeframe trend; volume confirms conviction
# Works in both bull and bear markets by following the 1w trend while using 6h for entry timing

name = "6h_ElderRay_BullBearPower_1wEMA50_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for trend filter and Elder Ray calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 on 1w close for Elder Ray (standard period)
    ema_13_1w = pd.Series(df_1w['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Bull Power and Bear Power on 1w
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power_1w = df_1w['high'].values - ema_13_1w
    bear_power_1w = df_1w['low'].values - ema_13_1w
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to 6h timeframe (wait for 1w bar to close)
    bull_power_1w_aligned = align_htf_to_ltf(prices, df_1w, bull_power_1w)
    bear_power_1w_aligned = align_htf_to_ltf(prices, df_1w, bear_power_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
        if (np.isnan(bull_power_1w_aligned[i]) or 
            np.isnan(bear_power_1w_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND 1w close > 1w EMA50 AND volume spike
            if (bull_power_1w_aligned[i] > 0 and 
                bear_power_1w_aligned[i] < 0 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power < 0 AND 1w close < 1w EMA50 AND volume spike
            elif (bear_power_1w_aligned[i] < 0 and 
                  bull_power_1w_aligned[i] < 0 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses below 0 (loss of bullish momentum)
            if bull_power_1w_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses above 0 (loss of bearish momentum)
            if bear_power_1w_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals