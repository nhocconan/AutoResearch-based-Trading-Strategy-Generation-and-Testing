#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Elder Ray (bull/bear power) with 1d EMA34 trend filter and volume confirmation
# - Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0, price > 1d EMA34, and volume expansion
# - Short when Bear Power > 0, price < 1d EMA34, and volume expansion
# - Exit when Elder Power signal reverses or price crosses 1d EMA34
# - Volume filter: current volume > 1.5x 20-period average
# - Designed to capture momentum in both bull and bear markets by measuring buying/selling pressure relative to trend
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_ElderRay_1dEMA34_Volume"
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
    
    # Get 12h data for Elder Ray calculation (13-period EMA)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_12h = high_12h - ema_13_12h  # Buying power
    bear_power_12h = ema_13_12h - low_12h   # Selling power
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_6h = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema_34_1d_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long when buying pressure > 0, above 1d EMA34, and volume expansion
            if bull_power_6h[i] > 0 and close[i] > ema_34_1d_6h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when selling pressure > 0, below 1d EMA34, and volume expansion
            elif bear_power_6h[i] > 0 and close[i] < ema_34_1d_6h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when selling pressure appears or price breaks below 1d EMA34
            if bear_power_6h[i] > 0 or close[i] < ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when buying pressure appears or price breaks above 1d EMA34
            if bull_power_6h[i] > 0 or close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals