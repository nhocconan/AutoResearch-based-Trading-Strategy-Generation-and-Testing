# 4h_Camarilla_R1S1_Breakout_Volume_Strict
# Hypothesis: Camarilla pivot levels (R1/S1) act as strong support/resistance in BTC/ETH.
# In bull markets, price breaks above R1 and continues up; in bear markets, breaks below S1 and continues down.
# Volume confirmation ensures breakout is genuine, not a false signal.
# Strict volume threshold (>1.5x 20-period average) reduces false breakouts and controls trade frequency.
# Target: 20-40 trades/year per symbol to avoid fee drag.
# Works in both bull (breakout long) and bear (breakout short) markets.

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    rng = high_1d - low_1d
    r1_1d = close_1d + 1.1 * rng / 12
    s1_1d = close_1d - 1.1 * rng / 12
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count > 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close breaks above R1 with volume confirmation
            if close[i] > r1_4h[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S1 with volume confirmation
            elif close[i] < s1_4h[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close breaks below S1 (reversal signal)
            if close[i] < s1_4h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close breaks above R1 (reversal signal)
            if close[i] > r1_4h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Strict"
timeframe = "4h"
leverage = 1.0