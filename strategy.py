#!/usr/bin/env python3
"""
12h Camarilla pivot reversal with 1w trend filter and volume confirmation
Hypothesis: Price rejecting Camarilla pivot levels (S3/S4 for long, R3/R4 for short) 
in alignment with 1-week EMA(50) trend, confirmed by volume surge (current volume > 2.0x 
20-period average), captures reversals in both bull and bear markets. Uses 1w trend filter 
to avoid counter-trend trades and volume confirmation to ensure momentum. Target: 15-35 
trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_reversal_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_surge[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR price reaches Camarilla R3 level
            if i >= 20:  # Need 20 days for pivot calculation
                # Calculate Camarilla levels from previous day's range
                phigh = high[i-1]
                plow = low[i-1]
                pclose = close[i-1]
                range_val = phigh - plow
                r3 = pclose + (range_val * 1.1000 / 4)  # R3 level
                
                if (close[i] <= ema_50_1w_aligned[i] or 
                    close[i] >= r3):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.0  # Not enough data for pivot calculation
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR price reaches Camarilla S3 level
            if i >= 20:
                # Calculate Camarilla levels from previous day's range
                phigh = high[i-1]
                plow = low[i-1]
                pclose = close[i-1]
                range_val = phigh - plow
                s3 = pclose - (range_val * 1.1000 / 4)  # S3 level
                
                if (close[i] >= ema_50_1w_aligned[i] or 
                    close[i] <= s3):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0  # Not enough data for pivot calculation
                
        else:  # Flat, look for entry
            if i >= 20:  # Need 20 days for pivot calculation
                # Calculate Camarilla levels from previous day's range
                phigh = high[i-1]
                plow = low[i-1]
                pclose = close[i-1]
                range_val = phigh - plow
                
                # Camarilla levels
                r3 = pclose + (range_val * 1.1000 / 4)  # R3 level
                r4 = pclose + (range_val * 1.1000 / 2)  # R4 level
                s3 = pclose - (range_val * 1.1000 / 4)  # S3 level
                s4 = pclose - (range_val * 1.1000 / 2)  # S4 level
                
                # Long: price rejects S3/S4 level + volume surge + uptrend
                if ((close[i] <= s3 or close[i] <= s4) and
                    close[i] > ema_50_1w_aligned[i] and
                    vol_surge[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price rejects R3/R4 level + volume surge + downtrend
                elif ((close[i] >= r3 or close[i] >= r4) and
                      close[i] < ema_50_1w_aligned[i] and
                      vol_surge[i]):
                    position = -1
                    signals[i] = -0.25
            # Not enough data for pivot calculation
    
    return signals