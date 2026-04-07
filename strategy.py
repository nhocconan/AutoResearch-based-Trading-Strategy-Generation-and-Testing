#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from 1-day data with volume confirmation
# Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels
# Uses 1-day Camarilla pivots calculated from previous day's OHLC
# Long when price crosses above S3 with volume > 1.5x 6h average volume, targeting S4
# Short when price crosses below R3 with volume > 1.5x 6h average volume, targeting R4
# Exit when price reaches target level (S4 for longs, R4 for shorts) or reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_camarilla_1d_vol_v1"
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
    volume = volumes = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: 
    # Resistance: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # Support: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    rng = high_1d - low_1d
    r4 = close_1d + rng * 1.1 / 2
    r3 = close_1d + rng * 1.1 / 4
    s3 = close_1d - rng * 1.1 / 4
    s4 = close_1d - rng * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h volume average for confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    target_level = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_ma_6h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                target_level = 0.0
            # Exit: reached target (S4) or reversed below S3
            elif close[i] >= s4_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                target_level = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                target_level = 0.0
            # Exit: reached target (R4) or reversed above R3
            elif close[i] <= r4_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                target_level = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price crosses above S3 with volume spike
            if (close[i] > s3_aligned[i] and 
                close[i-1] <= s3_aligned[i-1] and
                volume[i] > 1.5 * volume_ma_6h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                target_level = s4_aligned[i]
            # Short: price crosses below R3 with volume spike
            elif (close[i] < r3_aligned[i] and 
                  close[i-1] >= r3_aligned[i-1] and
                  volume[i] > 1.5 * volume_ma_6h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                target_level = r4_aligned[i]
    
    return signals