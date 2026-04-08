#!/usr/bin/env python3
# 6h_parabolic_sar_volume_v1
# Hypothesis: Parabolic SAR identifies trend direction on 6h timeframe. Volume confirmation filters false breakouts.
# Works in bull/bear by following trend with trailing stop. Volume surge required for entry.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_parabolic_sar_volume_v1"
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
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize arrays
    psar = np.full(n, np.nan)
    bull = np.full(n, True)  # True = bullish trend
    af = np.full(n, af_start)
    ep = np.full(n, np.nan)  # Extreme point
    
    # Set initial values
    psar[0] = low[0]
    ep[0] = high[0]
    
    # Calculate PSAR
    for i in range(1, n):
        if bull[i-1]:  # Was bullish
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for reversal
            if low[i] <= psar[i]:
                bull[i] = False
                psar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = low[i]
                af[i] = af_start
            else:
                bull[i] = True
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Was bearish
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for reversal
            if high[i] >= psar[i]:
                bull[i] = True
                psar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = high[i]
                af[i] = af_start
            else:
                bull[i] = False
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 1  # Start from second bar
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(psar[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below PSAR or volume drops below average
            if close[i] <= psar[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above PSAR or volume drops below average
            if close[i] >= psar[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above PSAR, volume surge
            if close[i] > psar[i] and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below PSAR, volume surge
            elif close[i] < psar[i] and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals