#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13 (buying strength), Bear Power = EMA13 - Low (selling strength).
# In bull markets: Buy when Bull Power > 0 and rising, price > EMA34, volume spike.
# In bear markets: Sell when Bear Power > 0 and rising, price < EMA34, volume spike.
# Elder Ray captures institutional buying/selling pressure; EMA34 filter avoids counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray (13-period EMA of close)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # Buying strength
    bear_power_1d = ema13_1d - low_1d   # Selling strength
    
    # Align Elder Ray components to 6h timeframe (wait for 1d bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Rising momentum: current value > previous value
    bull_power_rising = bull_power_aligned > np.roll(bull_power_aligned, 1)
    bear_power_rising = bear_power_aligned > np.roll(bear_power_aligned, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: Bull Power > 0 and rising, price > EMA34, volume spike
        if (bull_power_aligned[i] > 0 and 
            bull_power_rising[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short entry: Bear Power > 0 and rising, price < EMA34, volume spike
        elif (bear_power_aligned[i] > 0 and 
              bear_power_rising[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or loss of momentum
        elif position == 1 and (bull_power_aligned[i] <= 0 or not bull_power_rising[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bear_power_aligned[i] <= 0 or not bear_power_rising[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0