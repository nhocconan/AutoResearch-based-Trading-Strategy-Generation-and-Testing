#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot (from daily) with volume confirmation.
# Camarilla levels provide intraday support/resistance: fade at R3/S3, breakout continuation at R4/S4.
# Uses daily pivot from previous day (no look-ahead) to calculate Camarilla levels.
# Volume filter requires current volume > 1.3x 20-period average to avoid false signals.
# Works in bull/bear markets via mean reversion at R3/S3 and breakout continuation at R4/S4.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

name = "6h_camarilla_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day's OHLC
    # Use shifted values to avoid look-ahead: yesterday's data for today's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels (based on previous day's range)
    r3 = pivot + range_hl * 1.1 / 4
    s3 = pivot - range_hl * 1.1 / 4
    r4 = pivot + range_hl * 1.1 / 2
    s4 = pivot - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (already shift(1) via align_htf_to_ltf)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Price levels
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        r4_val = r4_6h[i]
        s4_val = s4_6h[i]
        
        # Check exits and stoploss (using simple range-based stop)
        if position == 1:  # long position
            # Exit conditions: mean reversion at R3 or stoploss
            atr_approx = (high[i] - low[i]) if (high[i] - low[i]) > 0 else 0.001
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] >= r3_val and volume_filter) or close[i] < stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions: mean reversion at S3 or stoploss
            atr_approx = (high[i] - low[i]) if (high[i] - low[i]) > 0 else 0.001
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] <= s3_val and volume_filter) or close[i] > stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Fade at R3/S3: sell at R3, buy at S3 (mean reversion)
                if close[i] <= s3_val and close[i] > s4_val:
                    # Long near S3 with rejection from below
                    if i > 0 and close[i] > close[i-1]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                elif close[i] >= r3_val and close[i] < r4_val:
                    # Short near R3 with rejection from above
                    if i > 0 and close[i] < close[i-1]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                # Breakout continuation at R4/S4
                elif close[i] > r4_val:
                    # Break above R4 - go long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < s4_val:
                    # Break below S4 - go short
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals