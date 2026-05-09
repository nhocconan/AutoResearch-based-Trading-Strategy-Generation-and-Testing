#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation
# Long when: close > Donchian upper(20), ATR(14) > 1.5x 20-period ATR MA, volume > 1.5x 20-period volume MA
# Short when: close < Donchian lower(20), ATR(14) > 1.5x 20-period ATR MA, volume > 1.5x 20-period volume MA
# Exit when: price crosses Donchian midpoint OR volatility drops below threshold
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 25-50 trades/year.
# Works in trending markets (breakouts) and avoids low-volatility chop.

name = "4h_Donchian20_ATRVolFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # ATR(14) for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR volatility filter: current ATR > 1.5x 20-period ATR MA
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > (1.5 * atr_ma)
    
    # Volume confirmation: current volume > 1.5x 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(vol_filter[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Donchian upper + volatility filter + volume spike
            if (close[i] > donchian_upper[i] and 
                vol_filter[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian lower + volatility filter + volume spike
            elif (close[i] < donchian_lower[i] and 
                  vol_filter[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR volatility drops
            if (close[i] < donchian_mid[i]) or (not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR volatility drops
            if (close[i] > donchian_mid[i]) or (not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals