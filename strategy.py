#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation and ATR filter.
# Long when price breaks above R4 AND volume > 1.5x 20-period average AND ATR(14) < 0.03 * price (low volatility breakout).
# Short when price breaks below S4 AND volume > 1.5x 20-period average AND ATR(14) < 0.03 * price.
# Exit when price retests the 1d pivot point (PP) or reverses to opposite Camarilla level (R3/S3).
# Uses discrete position size 0.25. 1d Camarilla provides structure, 6h provides execution timing.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, R4, S3, S4, PP) ===
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = close_1d + range_1d * 1.1 / 2
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # === 1d Indicators: ATR (14) for volatility filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (alpha=1/14)
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (6h)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        pp = pp_aligned[i]
        r4 = r4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 6h volume average aligned
        vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
        # Volatility filter: ATR < 3% of price (low volatility breakout)
        vol_filter = atr < 0.03 * price
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price <= pp or price >= r3:  # Exit when price retests PP or reaches R3
                exit_signal = True
        
        elif position == -1:  # Short position
            if price >= pp or price <= s3:  # Exit when price retests PP or reaches S3
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R4 AND volume > 1.5x 20-period avg AND low volatility
            if (price > r4) and (vol > 1.5 * vol_ma_20_6h[i]) and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S4 AND volume > 1.5x 20-period avg AND low volatility
            elif (price < s4) and (vol > 1.5 * vol_ma_20_6h[i]) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dCamarillaR4S4_Volume_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0