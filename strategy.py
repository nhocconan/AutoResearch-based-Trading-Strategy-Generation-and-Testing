#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume spike.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# We go long when Bull Power > 0 and Bear Power < 0 (both bullish) with 1d EMA34 uptrend and volume spike.
# Short when Bear Power > 0 and Bull Power < 0 (both bearish) with 1d EMA34 downtrend and volume spike.
# Uses volume > 2x average to avoid false signals in low volume.
# Designed to work in both bull and bear markets by requiring trend alignment.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_ElderRay_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Calculate volume average (20-period) for spike detection
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 34 for EMA and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol = volume[i]
        vol_avg = vol_ma[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 AND Bear Power < 0 (both bullish) AND price > 1d EMA34 (uptrend) AND volume > 2x average
            if bp > 0 and br < 0 and close[i] > ema_1d and vol > 2.0 * vol_avg:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power > 0 AND Bull Power < 0 (both bearish) AND price < 1d EMA34 (downtrend) AND volume > 2x average
            elif br > 0 and bp < 0 and close[i] < ema_1d and vol > 2.0 * vol_avg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 (loss of bullish momentum) OR trend reverses (price < 1d EMA34)
            if bp <= 0 or br >= 0 or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power <= 0 OR Bull Power >= 0 (loss of bearish momentum) OR trend reverses (price > 1d EMA34)
            if br <= 0 or bp >= 0 or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals