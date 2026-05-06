#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and chop filter
# Long when price breaks above weekly Donchian upper channel (20-period high) with volume > 1.5x 20-period average and chop > 61.8 (range)
# Short when price breaks below weekly Donchian lower channel (20-period low) with volume > 1.5x 20-period average and chop > 61.8 (range)
# Uses weekly Donchian for key support/resistance, volume for breakout confirmation, chop filter to avoid whipsaws in strong trends
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing

name = "1d_weeklyDonchian20_Volume_Chop_v1"
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
    
    # Calculate weekly Donchian Channel (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period high and low for Donchian channels
    high_20 = df_1w['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1w, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Chop filter: avoid strong trends, prefer ranging markets
    atr_period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10(highest_high - lowest_low) / np.log10(atr_period) / np.log10(np.sum(tr, axis=0, keepdims=True).flatten()) if False else \
           100 * np.log10((highest_high - lowest_low) / (atr * atr_period)) / np.log10(atr_period)
    chop = np.where((highest_high - lowest_low) > 0, 100 * np.log10((highest_high - lowest_low) / (atr * atr_period)) / np.log10(atr_period), 50)
    chop_filter = chop > 61.8  # Range market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any critical value is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian upper with volume and chop confirmation
            if close[i] > upper_donchian[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian lower with volume and chop confirmation
            elif close[i] < lower_donchian[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian lower (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian upper (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals