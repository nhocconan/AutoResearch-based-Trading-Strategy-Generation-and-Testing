#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
Long when price breaks above R4 with volume confirmation (bullish continuation).
Short when price breaks below S4 with volume confirmation (bearish continuation).
Otherwise, fade at R3/S3 levels with volume divergence (mean reversion in ranging markets).
Uses stricter volume confirmation and cooldown periods to reduce trade frequency.
Works in both bull/bear markets by adapting to volatility and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    # Avoid division by zero
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    
    # Camarilla levels
    r3 = prev_close + range_1d * 1.1 / 4
    r4 = prev_close + range_1d * 1.1 / 2
    s3 = prev_close - range_1d * 1.1 / 4
    s4 = prev_close - range_1d * 1.1 / 2
    
    # Align to 12h timeframe (previous day's levels are valid for current day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 40-period average (stricter)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    # Cooldown counter to prevent overtrading
    cooldown = 0
    cooldown_period = 4  # 4 bars = 48 hours minimum between trades
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Decrease cooldown
        if cooldown > 0:
            cooldown -= 1
        
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 (mean reversion) or stop at S4 break
            if close[i] < r3_aligned[i]:
                position = 0
                signals[i] = 0.0
                cooldown = cooldown_period
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (mean reversion) or stop at R4 break
            if close[i] > s3_aligned[i]:
                position = 0
                signals[i] = 0.0
                cooldown = cooldown_period
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry (only if cooldown is 0)
            if cooldown > 0:
                signals[i] = 0.0
                continue
                
            # Breakout long: price breaks above R4 with volume
            if close[i] > r4_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
                cooldown = cooldown_period
            # Breakout short: price breaks below S4 with volume
            elif close[i] < s4_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
                cooldown = cooldown_period
            # Mean reversion long: price rejects S3 with volume divergence (lower volume on test)
            elif close[i] < s3_aligned[i] and volume[i] < vol_ma[i] * 0.7:
                # Look for bullish rejection (close > open)
                if close[i] > prices['open'].iloc[i]:
                    position = 1
                    signals[i] = 0.25
                    cooldown = cooldown_period
            # Mean reversion short: price rejects R3 with volume divergence
            elif close[i] > r3_aligned[i] and volume[i] < vol_ma[i] * 0.7:
                # Look for bearish rejection (close < open)
                if close[i] < prices['open'].iloc[i]:
                    position = -1
                    signals[i] = -0.25
                    cooldown = cooldown_period
    
    return signals