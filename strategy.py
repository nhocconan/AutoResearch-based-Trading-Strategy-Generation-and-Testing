#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_With_Volume
Hypothesis: Camarilla pivot levels (S3/S4 for shorts, R3/R4 for longs) on 1d act as
strong support/resistance. Price touching these levels with volume confirmation
and mean-reversion signals (RSI extreme) provides high-probability reversals.
Works in both bull and bear markets as reversals occur at key levels regardless
of trend. Uses 12h for entry timing and 1d for pivot calculation.
Target: 15-30 trades/year per symbol.
"""

name = "12h_Camarilla_Pivot_Reversal_With_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # RSI(14) for overbought/oversold signals
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate Camarilla pivot levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC to calculate today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_1d * 1.500
    r3 = close_1d + range_1d * 1.250
    s3 = close_1d - range_1d * 1.250
    s4 = close_1d - range_1d * 1.500
    
    # Align 1d levels to 12h (these levels are valid for the entire 1d period)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (20-period average)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or
            np.isnan(r4_12h[i]) or np.isnan(r3_12h[i]) or
            np.isnan(s3_12h[i]) or np.isnan(s4_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Long setup: price touches/slightly penetrates S3/S4 with RSI oversold and volume
            touches_s3 = low[i] <= s3_12h[i] * 1.002  # Allow 0.2% slippage
            touches_s4 = low[i] <= s4_12h[i] * 1.002
            rsi_oversold = rsi[i] < 30
            
            if ((touches_s3 or touches_s4) and rsi_oversold and volume_confirm):
                signals[i] = 0.25
                position = 1
            
            # Short setup: price touches/slightly penetrates R3/R4 with RSI overbought and volume
            elif (high[i] >= r3_12h[i] * 0.998 or high[i] >= r4_12h[i] * 0.998) and \
                 rsi[i] > 70 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or price reaches pivot
            if rsi[i] >= 50 or high[i] >= pivot[i] * 0.999:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or price reaches pivot
            if rsi[i] <= 50 or low[i] <= pivot[i] * 1.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals