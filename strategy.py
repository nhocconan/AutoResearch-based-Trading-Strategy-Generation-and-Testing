#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP)
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Calculate Camarilla levels
    # Resistance levels
    r1_12h = close_12h + ((high_12h - low_12h) * 1.1 / 12)
    r2_12h = close_12h + ((high_12h - low_12h) * 1.1 / 6)
    r3_12h = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    r4_12h = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    # Support levels
    s1_12h = close_12h - ((high_12h - low_12h) * 1.1 / 12)
    s2_12h = close_12h - ((high_12h - low_12h) * 1.1 / 6)
    s3_12h = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    s4_12h = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align 12h Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Long entry: price breaks above R3 with volume confirmation
        long_signal = price_high > r3_aligned[i] and volume_confirmed
        
        # Short entry: price breaks below S3 with volume confirmation
        short_signal = price_low < s3_aligned[i] and volume_confirmed
        
        # Exit conditions
        exit_long = position == 1 and price_close < r1_aligned[i]
        exit_short = position == -1 and price_close > s1_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.30
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla breakout strategy on 4h timeframe using 12h pivot levels.
# Uses 12h Camarilla levels (R3/S3 for entry, R1/S1 for exit) to identify institutional support/resistance.
# Enters long when price breaks above R3 with volume confirmation (>1.3x average volume).
# Enters short when price breaks below S3 with volume confirmation.
# Exits when price returns to R1/S1 levels respectively.
# Works in both bull and bear markets by trading breakouts in either direction from key levels.
# Volume confirmation reduces false breakouts. Target: 20-40 trades/year to minimize fee drag.
# Based on top-performing Camarilla strategies from the database.