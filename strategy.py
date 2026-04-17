#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 1d volume spike filter and ATR-based trailing stop.
Long when price breaks above Camarilla R3 level AND 1d volume > 2.0x 20-day average.
Short when price breaks below Camarilla S3 level AND 1d volume > 2.0x 20-day average.
Exit when price reverts to Camarilla H3/L3 levels OR ATR trailing stop hit (2.5x ATR from extreme).
Uses 4h for Camarilla calculation (derived from 1d OHLC) and 1d for volume filter to avoid lower-timeframe noise.
Trailing stop reduces whipsaw in ranging markets while capturing trends. Target: 75-200 total trades over 4 years.
Camarilla levels work well in both bull and bear markets as they adapt to recent volatility.
Volume spike confirms institutional interest, reducing false breakouts.
ATR trailing stop allows profits to run while limiting drawdowns.
"""

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
    
    # Get 1d data for Camarilla calculation (based on previous day OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # We use R3/S3 for entry and H3/L3 for exit (more conservative)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + 1.125 * rng
    camarilla_s3 = close_1d - 1.125 * rng
    camarilla_h3 = close_1d + 1.000 * rng  # H3/L3 for exit
    camarilla_l3 = close_1d - 1.000 * rng
    
    # Align 1d Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 1d data for volume filter
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # ATR for trailing stop (using 1d ATR for consistency with volume filter)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR (14-period)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    long_high = 0.0   # highest high since entering long
    short_low = 0.0   # lowest low since entering short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume_1d[i]  # use 1d volume for filter
        atr_val = atr_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R3 AND volume > 2.0x 20-day average
            if price > r3 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                long_high = price  # initialize trailing stop high
            # Short: price < Camarilla S3 AND volume > 2.0x 20-day average
            elif price < s3 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                short_low = price  # initialize trailing stop low
        
        elif position == 1:
            # Update trailing stop high
            long_high = max(long_high, price)
            # Exit long: price < Camarilla H3 OR price drops 2.5*ATR from high
            if price < h3 or price < long_high - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update trailing stop low
            short_low = min(short_low, price)
            # Exit short: price > Camarilla L3 OR price rises 2.5*ATR from low
            if price > l3 or price > short_low + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CamarillaR3S3_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0