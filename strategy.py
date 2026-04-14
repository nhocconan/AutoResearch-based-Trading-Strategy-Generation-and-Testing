#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 12h Trend Filter and Volume Spike
# Uses Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) to measure bull/bear strength
# Long when Bull Power > 0 and rising + price above 12h EMA20 + volume spike
# Short when Bear Power > 0 and rising + price below 12h EMA20 + volume spike
# Uses 12h EMA20 as trend filter to avoid counter-trend trades
# Volume spike filter (1.5x average) ensures participation during strong moves
# Designed to work in both bull (strong bull power) and bear (strong bear power) markets
# Target: 60-120 trades over 4 years (15-30/year)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA13 for Elder Ray (using close prices)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate volume average (20-period) for spike detection
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Elder Ray smoothing (13-period) to detect rising power
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Detect rising power (current > previous)
    bull_power_rising = bull_power_smooth > np.roll(bull_power_smooth, 1)
    bear_power_rising = bear_power_smooth > np.roll(bear_power_smooth, 1)
    # Handle first element
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: Bull Power > 0 and rising + price above 12h EMA20 + volume spike
            if (bull_power[i] > 0 and bull_power_rising[i] and 
                price > ema20_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                position = 1
                signals[i] = position_size
            # Short setup: Bear Power > 0 and rising + price below 12h EMA20 + volume spike
            elif (bear_power[i] > 0 and bear_power_rising[i] and 
                  price < ema20_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power becomes negative or price crosses below 12h EMA20
            if bull_power[i] <= 0 or price < ema20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power becomes negative or price crosses above 12h EMA20
            if bear_power[i] <= 0 or price > ema20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_12hEMA_VolumeSpike"
timeframe = "6h"
leverage = 1.0