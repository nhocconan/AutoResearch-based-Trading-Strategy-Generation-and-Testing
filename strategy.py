#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Uses Choppiness Index to identify trending vs ranging markets (avoid false breakouts in chop)
# Donchian breakouts provide clear entry/exit levels, volume confirms momentum
# Designed for low-frequency trades (<100 total) to minimize fee drag in choppy markets

name = "4h_Chop_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros(len(high))
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        atr[0] = np.nan
        
        # Sum of true range over period
        atr_sum = np.convolve(atr, np.ones(period), mode='same')
        atr_sum[:period-1] = np.nan
        
        # Highest high and lowest low over period
        highest_high = np.zeros(len(high))
        lowest_low = np.zeros(len(low))
        for i in range(len(high)):
            start = max(0, i - period + 1)
            highest_high[i] = np.max(high[start:i+1])
            lowest_low[i] = np.min(low[start:i+1])
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        chop = np.where(range_hl != 0, 100 * np.log10(atr_sum / period) / np.log10(range_hl), 50)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.zeros(len(high))
        lower = np.zeros(len(low))
        for i in range(len(high)):
            start = max(0, i - period + 1)
            upper[i] = np.max(high[start:i+1])
            lower[i] = np.min(low[start:i+1])
        # Set NaN for insufficient data
        upper[:period-1] = np.nan
        lower[:period-1] = np.nan
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation (1.5x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when trending (CHOP < 38.2) or strong breakout in chop
        is_trending = chop[i] < 38.2
        is_strong_breakout = (close[i] > donch_upper[i] * 1.01 or close[i] < donch_lower[i] * 0.99)
        
        if position == 0:
            # Enter long: Donchian breakout up with volume and (trending OR strong breakout)
            if (close[i] > donch_upper[i] and vol_spike[i] and (is_trending or is_strong_breakout)):
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakdown down with volume and (trending OR strong breakout)
            elif (close[i] < donch_lower[i] and vol_spike[i] and (is_trending or is_strong_breakout)):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel or chop becomes extreme
            if (close[i] < donch_upper[i] or chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel or chop becomes extreme
            if (close[i] > donch_lower[i] or chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals