#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 12h EMA34 filter and volume spike confirmation.
# Long when: Bull Power > 0, Bear Power < 0, 12h EMA34 upward, volume > 1.5x 20-period average
# Short when: Bull Power < 0, Bear Power > 0, 12h EMA34 downward, volume > 1.5x 20-period average
# Exit when: Bull Power and Bear Power cross (Bull Power < Bear Power for longs, Bull Power > Bear Power for shorts)
# Elder Ray measures bull/bear power relative to EMA, 12h EMA34 filters trend, volume confirms strength.
# Target: 12-37 trades/year per symbol. Works in bull (buy strength) and bear (sell weakness).
name = "6h_ElderRay_EMA13_Volume"
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
    
    # 12-hour data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate EMA34 on 12h data for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34)  # Wait for EMA13 and EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        ema34 = ema34_12h_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Bull Power > 0, Bear Power < 0, 12h EMA34 upward, volume spike
            if (bp > 0 and br < 0 and 
                ema34 > ema34_12h_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Bull Power < 0, Bear Power > 0, 12h EMA34 downward, volume spike
            elif (bp < 0 and br > 0 and 
                  ema34 < ema34_12h_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power crosses below Bear Power (momentum weakening)
            if bp < br:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power crosses above Bear Power (momentum strengthening)
            if bp > br:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals