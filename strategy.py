#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and chop regime filter.
# Long when price breaks above R3 with volume > 1.5x 20-period average AND Chop(14) > 61.8 (ranging market).
# Short when price breaks below S3 with volume > 1.5x 20-period average AND Chop(14) > 61.8.
# Exit when price reaches opposite pivot level (S1 for longs, R1 for shorts) or crosses the pivot point (mean reversion).
# Uses discrete position size 0.25. 1d pivots provide structure, chop filter avoids trending markets where pivots fail.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels ===
    # Pivot point (PP) = (High + Low + Close) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1_1d = pp_1d + (range_1d * 1.1 / 12)
    r2_1d = pp_1d + (range_1d * 1.1 / 6)
    r3_1d = pp_1d + (range_1d * 1.1 / 4)
    r4_1d = pp_1d + (range_1d * 1.1 / 2)
    
    # Support levels
    s1_1d = pp_1d - (range_1d * 1.1 / 12)
    s2_1d = pp_1d - (range_1d * 1.1 / 6)
    s3_1d = pp_1d - (range_1d * 1.1 / 4)
    s4_1d = pp_1d - (range_1d * 1.1 / 2)
    
    # Align all pivot levels to primary timeframe (12h)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 12h Indicators: Chopiness Index (14) for regime filter ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) for denominator
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Chopiness Index = 100 * log10(sum(TR14) / (max_high - min_low)) / log10(14)
    # We'll compute it as: 100 * log10(rolling_sum(tr,14) / (rolling_max(high,14) - rolling_min(low,14))) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(tr_sum / range_14) / np.log10(14)
    
    # Volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        pp = pp_aligned[i]
        r1 = r1_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s1 = s1_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        chop_val = chop[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reaches S1 (profit target) or crosses below pivot (mean reversion)
            if price <= s1 or price < pp:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reaches R1 (profit target) or crosses above pivot (mean reversion)
            if price >= r1 or price > pp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Regime filter: only trade in ranging markets (Chop > 61.8)
            regime_filter = chop_val > 61.8
            
            # Volume confirmation: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: Price breaks above R3 with volume confirmation and ranging market
            if (price > r3) and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S3 with volume confirmation and ranging market
            elif (price < s3) and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dCamarillaR3S3_ChopVol_V1"
timeframe = "12h"
leverage = 1.0