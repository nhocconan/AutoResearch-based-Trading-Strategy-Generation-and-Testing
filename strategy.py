#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Donchian breakout with volume confirmation and chop filter
# Long when price breaks above 1-week Donchian upper channel (20-period high) with volume > 1.5x 20-period average and chop > 61.8 (range)
# Short when price breaks below 1-week Donchian lower channel (20-period low) with volume > 1.5x 20-period average and chop > 61.8 (range)
# Uses weekly Donchian for structure, volume for breakout strength, chop for range filtering
# Designed to work in both bull/bear via breakouts and in range via mean reversion at extremes
# Target: 10-25 trades per year (40-100 over 4 years) with 0.30 position sizing

name = "1d_1wDonchian20_Volume_Chop_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-week Donchian Channel (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period high and low for Donchian channels
    high_20 = df_1w['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1w, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Chop filter: Chop > 61.8 indicates range (mean reversion zone)
    atr_period = 14
    high_low = pd.Series(high - low)
    atr = high_low.rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10(atr * atr_period / (highest_high - lowest_low)) / np.log10(atr_period)
    chop_filter = chop > 61.8  # Range condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume confirmation in range
            if close[i] > upper_donchian[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short breakout: price breaks below lower Donchian with volume confirmation in range
            elif close[i] < lower_donchian[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above upper Donchian (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals