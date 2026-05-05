#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Bull Power = High - EMA13(1d), Bear Power = Low - EMA13(1d)
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND close > EMA34(1d) AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND close < EMA34(1d) AND volume > 1.5x 20-period average
# Exit when Bull/Bear Power crosses zero OR EMA34(1d) trend flips
# Elder Ray measures bull/bear strength relative to 1d EMA13, providing confluence with 1d EMA34 trend filter
# Volume confirmation ensures institutional participation
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for Elder Ray and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d close for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Bull Power and Bear Power for 1d
    bull_power = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power rising (less negative) AND close > EMA34(1d) AND volume spike
            bull_power_rising = (i > 50 and bear_power_aligned[i] > bear_power_aligned[i-1])
            if (bull_power_aligned[i] > 0 and 
                bull_power_rising and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power falling (less positive) AND close < EMA34(1d) AND volume spike
            elif (bear_power_aligned[i] < 0 and 
                  (i > 50 and bull_power_aligned[i] < bull_power_aligned[i-1]) and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses below zero OR close < EMA34(1d) (trend flip)
            if (bull_power_aligned[i] <= 0 or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses above zero OR close > EMA34(1d) (trend flip)
            if (bear_power_aligned[i] >= 0 or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals