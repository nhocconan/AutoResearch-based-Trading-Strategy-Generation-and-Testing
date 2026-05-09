#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price above 1d EMA50
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising, price below 1d EMA50
# Volume confirmation reduces false signals. Targets 15-35 trades/year.
# Works in bull/bear by requiring alignment with 1d trend.
name = "6h_ElderRay_1dEMA50_Trend_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA
    bear_power = low - ema_13   # Low - EMA
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA and volume calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] < bear_power[i-1]
        bull_power_negative = bull_power[i] < 0
        bear_power_negative = bear_power[i] < 0
        bull_power_rising_from_neg = bull_power[i] > bull_power[i-1] and bull_power[i] < 0
        bear_power_falling_from_pos = bear_power[i] < bear_power[i-1] and bear_power[i] > 0
        
        if position == 0:
            # Long: Bull Power rising from negative + price above 1d EMA50 + volume
            if bull_power_rising_from_neg and close[i] > ema_50_6h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power falling from positive + price below 1d EMA50 + volume
            elif bear_power_falling_from_pos and close[i] < ema_50_6h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative or price breaks below 1d EMA50
            if bull_power[i] < 0 or close[i] < ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive or price breaks above 1d EMA50
            if bear_power[i] > 0 or close[i] > ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals