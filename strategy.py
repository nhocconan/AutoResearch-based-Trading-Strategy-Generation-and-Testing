#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# In bull trend (price > 1-day EMA50): buy when Bull Power > 0 and rising
# In bear trend (price < 1-day EMA50): sell when Bear Power < 0 and falling
# Volume confirmation: require volume > 1.3x 20-period average
# Designed to capture institutional buying/selling pressure in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema13  # Bull Power: High - EMA(13)
    bear_power = low - ema13   # Bear Power: Low - EMA(13)
    
    # Calculate volume filter: volume > 1.3x 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend
        is_bull = close[i] > ema50_1d_aligned[i]
        is_bear = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long conditions: Bull Power positive and rising in bull trend
            long_signal = False
            if has_volume and is_bull:
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    long_signal = True
            
            # Enter short conditions: Bear Power negative and falling in bear trend
            short_signal = False
            if has_volume and is_bear:
                if bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative or trend changes
            exit_signal = False
            if bull_power[i] <= 0 or not is_bull:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive or trend changes
            exit_signal = False
            if bear_power[i] >= 0 or not is_bear:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0