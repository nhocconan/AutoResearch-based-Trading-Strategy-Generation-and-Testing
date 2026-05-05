#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 AND 1d close > 1d EMA34 AND volume > 2.0x 20-period average
# Short when Bear Power < 0 AND 1d close < 1d EMA34 AND volume > 2.0x 20-period average
# Exit when Elder Power crosses zero (trend exhaustion)
# Uses 6h primary timeframe with 1d HTF for trend filter and Elder Ray calculation
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Elder Ray measures bull/bear strength relative to EMA; 1d EMA34 filters for higher-timeframe trend; volume confirms participation

name = "6h_Elder_Ray_BullBearPower_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray Bull Power and Bear Power on 1d data
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe (wait for 1d bar to close)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND 1d close > 1d EMA34 AND volume spike
            if (bull_power_1d_aligned[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND 1d close < 1d EMA34 AND volume spike
            elif (bear_power_1d_aligned[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses below zero (bullish exhaustion)
            if bull_power_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses above zero (bearish exhaustion)
            if bear_power_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals