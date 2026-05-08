#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and volume confirmation
# Elder Ray measures bullish/bearish power relative to EMA. Bull Power = High - EMA, Bear Power = Low - EMA.
# Strong bullish power indicates buying pressure, strong bearish power indicates selling pressure.
# We use 12h EMA for trend direction (long when Bull Power > 0 and EMA rising, short when Bear Power < 0 and EMA falling).
# Volume surge confirms institutional participation. This works in bull/bear by following the trend with momentum confirmation.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.

name = "6h_ElderRay_12hEMA_Volume"
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
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate EMA(21) on 12h
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate EMA(13) for Elder Ray on 6h (using close prices)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = low - ema_13   # Bear Power = Low - EMA
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    # EMA trend filter: EMA rising/falling
    ema_rising = ema_12h_aligned > np.roll(ema_12h_aligned, 1)
    ema_falling = ema_12h_aligned < np.roll(ema_12h_aligned, 1)
    # Handle first element
    ema_rising[0] = False
    ema_falling[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, EMA rising, volume surge
            if bull_power[i] > 0 and ema_rising[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, EMA falling, volume surge
            elif bear_power[i] < 0 and ema_falling[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or EMA not rising
            if bull_power[i] <= 0 or not ema_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or EMA not falling
            if bear_power[i] >= 0 or not ema_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals