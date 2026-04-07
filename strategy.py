#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_volume_v1
Hypothesis: Weekly Camarilla pivot levels act as strong support/resistance. 
Long when price breaks above weekly R4 with volume confirmation (bullish continuation).
Short when price breaks below weekly S4 with volume confirmation (bearish continuation).
Otherwise, fade at weekly R3/S3 levels with volume divergence (mean reversion in ranging markets).
Uses daily timeframe for entries with weekly pivot context to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values (shifted by 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan  # First week has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    range_1w = prev_high - prev_low
    # Avoid division by zero
    range_1w = np.where(range_1w == 0, 1e-10, range_1w)
    
    # Camarilla levels (using 1.1 multiplier as per standard formula)
    r3 = prev_close + range_1w * 1.1 / 4
    r4 = prev_close + range_1w * 1.1 / 2
    s3 = prev_close - range_1w * 1.1 / 4
    s4 = prev_close - range_1w * 1.1 / 2
    
    # Align to daily timeframe (previous week's levels are valid for current week)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume confirmation: volume > 20-period average (daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 (mean reversion) or stop at S4 break
            if close[i] < r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (mean reversion) or stop at R4 break
            if close[i] > s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long: price breaks above R4 with volume
            if close[i] > r4_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Breakout short: price breaks below S4 with volume
            elif close[i] < s4_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
            # Mean reversion long: price rejects S3 with volume divergence (lower volume on test)
            elif close[i] < s3_aligned[i] and volume[i] < vol_ma[i] * 0.8:
                # Look for bullish rejection (close > open)
                if close[i] > open_price[i]:
                    position = 1
                    signals[i] = 0.25
            # Mean reversion short: price rejects R3 with volume divergence
            elif close[i] > r3_aligned[i] and volume[i] < vol_ma[i] * 0.8:
                # Look for bearish rejection (close < open)
                if close[i] < open_price[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals