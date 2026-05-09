#!/usr/bin/env python3
# Hypothesis: 4h KAMA trend direction with 1d Bollinger Band squeeze and volume confirmation
# Long when: KAMA rising, price > upper BB(20,2) on 1d, volume spike (>1.5x 20-period avg)
# Short when: KAMA falling, price < lower BB(20,2) on 1d, volume spike
# Exit when: price crosses KAMA OR BB middle
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 20-40 trades/year.
# Designed to capture trends in bull markets and avoid whipsaws in bear via BB squeeze filter.

name = "4h_KAMA_BB_Squeeze_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (4h)
    price_change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, price_change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    bb_middle = close_1d.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_middle_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising + price > upper BB + volume spike
            if (kama_rising[i] and 
                close[i] > bb_upper_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling + price < lower BB + volume spike
            elif (kama_falling[i] and 
                  close[i] < bb_lower_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA OR below BB middle
            if (close[i] < kama[i]) or (close[i] < bb_middle_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA OR above BB middle
            if (close[i] > kama[i]) or (close[i] > bb_middle_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals