#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_volume_v1
Hypothesis: On 6-hour timeframe, use Camarilla pivot levels from the 1-day timeframe for mean-reversion entries. 
Long when price touches S3 level with volume confirmation and RSI < 30; short when price touches R3 level with volume confirmation and RSI > 70.
Exit when price reaches the opposite pivot level (S1/R1) or mean price.
Designed for 15-25 trades/year to minimize fee decay while exploiting intraday mean reversion around institutional pivot levels.
Works in both bull/bear markets as Camarilla levels adapt to volatility and RSI filter avoids extreme momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first day uses same day close
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    
    # Resistance and support levels
    r1 = close_prev + (range_1d * 1.1 / 12)
    r2 = close_prev + (range_1d * 1.1 / 6)
    r3 = close_prev + (range_1d * 1.1 / 4)
    r4 = close_prev + (range_1d * 1.1 / 2)
    
    s1 = close_prev - (range_1d * 1.1 / 12)
    s2 = close_prev - (range_1d * 1.1 / 6)
    s3 = close_prev - (range_1d * 1.1 / 4)
    s4 = close_prev - (range_1d * 1.1 / 2)
    
    # Align all levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # RSI(14) for overbought/oversold filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 24-period average (4 days on 6h chart)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(24, 14), n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current volume > 1.5x average
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 or mean price
            if close[i] >= s1_aligned[i] or close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 or mean price
            if close[i] <= r1_aligned[i] or close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price touches or goes below S3 with RSI < 30 (oversold)
                if close[i] <= s3_aligned[i] and rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches or goes above R3 with RSI > 70 (overbought)
                elif close[i] >= r3_aligned[i] and rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals