#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA)
# - 1d EMA21 as trend filter: Bull Power > 0 + price > 1d EMA21 = long, Bear Power < 0 + price < 1d EMA21 = short
# - Volume confirmation: current volume > 1.5x 20-period average
# - Exit when power crosses zero or volume drops below average
# - Target: 60-100 total trades over 4 years (15-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 21-period EMA for trend filter
    ema21_1d = close_1d_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align to 6h timeframe
    ema13_6h = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema21_6h = align_htf_to_ltf(prices, df_1d, ema21_1d)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema21_6h[i]) or np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: Bull Power > 0, price > 1d EMA21, volume surge
            if bull_power_6h[i] > 0 and price > ema21_6h[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0, price < 1d EMA21, volume surge
            elif bear_power_6h[i] > 0 and price < ema21_6h[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power <= 0 or volume drops below average
            if bull_power_6h[i] <= 0 or vol < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power <= 0 or volume drops below average
            if bear_power_6h[i] <= 0 or vol < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dTrendFilter_Volume"
timeframe = "6h"
leverage = 1.0