#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Chaikin Oscillator with 1d EMA34 trend filter and volume confirmation.
# Chaikin Oscillator = EMA3(ADL) - EMA10(ADL), where ADL = Money Flow Volume cumulative.
# Long when Chaikin > 0 AND EMA13 rising AND price > 1d EMA34 AND volume > 1.5x 20-period average.
# Short when Chaikin < 0 AND EMA13 falling AND price < 1d EMA34 AND volume > 1.5x 20-period average.
# Exit when Chaikin crosses zero or volume filter fails.
# Designed for 12h timeframe with low trade frequency (target: 15-25/year) to avoid fee drag.
# Uses 1d EMA34 for trend filter and Chaikin Oscillator for institutional flow confirmation.
name = "12h_ChaikinOscillator_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Money Flow Volume = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    # Avoid division by zero
    hl_range = high - low
    mf_multiplier = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    mf_volume = mf_multiplier * volume
    
    # ADL = cumulative sum of Money Flow Volume
    adl = np.cumsum(mf_volume)
    
    # Chaikin Oscillator = EMA3(ADL) - EMA10(ADL)
    adl_series = pd.Series(adl)
    ema3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3_adl - ema10_adl
    
    # EMA13 for trend and zero-cross confirmation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_rising = ema13[1:] > ema13[:-1]
    ema13_falling = ema13[1:] < ema13[:-1]
    ema13_rising = np.concatenate([[False], ema13_rising])
    ema13_falling = np.concatenate([[False], ema13_falling])
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chaikin[i]) or np.isnan(ema13[i]) or np.isnan(ema13_rising[i]) or 
            np.isnan(ema13_falling[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Chaikin > 0, EMA13 rising, price > 1d EMA34, volume filter
            long_cond = (chaikin[i] > 0) and ema13_rising[i] and (close[i] > ema34_1d_aligned[i]) and volume_filter[i]
            # Short conditions: Chaikin < 0, EMA13 falling, price < 1d EMA34, volume filter
            short_cond = (chaikin[i] < 0) and ema13_falling[i] and (close[i] < ema34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Chaikin crosses below zero OR volume filter fails
            if chaikin[i] <= 0 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Chaikin crosses above zero OR volume filter fails
            if chaikin[i] >= 0 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals