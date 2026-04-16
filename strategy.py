#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation and ATR-based stop.
# Long when price breaks above R3 with volume > 1.5x 20-period average AND ATR(14) < 0.02 * price (low volatility environment).
# Short when price breaks below S3 with volume > 1.5x 20-period average AND ATR(14) < 0.02 * price.
# Exit when price reaches R4/S4 (profit target) or crosses the pivot point (mean reversion).
# Uses discrete position size 0.25. 1d pivots provide structure, 6h provides entry timing and volatility filter.
# Target: 80-120 total trades over 4 years (20-30/year) to balance edge and fee drag.

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
    
    # Align all pivot levels to primary timeframe (6h)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: ATR (14) for volatility filter ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (alpha = 1/14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        pp = pp_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        atr = atr_14[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reaches R4 (profit target) or crosses below pivot (mean reversion)
            if price >= r4 or price < pp:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reaches S4 (profit target) or crosses above pivot (mean reversion)
            if price <= s4 or price > pp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volatility filter: only trade when ATR < 2% of price (low volatility environment)
            vol_filter = atr < 0.02 * price
            
            # LONG: Price breaks above R3 with volume confirmation and low volatility
            if (price > r3) and (vol > 1.5 * vol_ma) and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S3 with volume confirmation and low volatility
            elif (price < s3) and (vol > 1.5 * vol_ma) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dCamarillaR3S3_Vol_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0