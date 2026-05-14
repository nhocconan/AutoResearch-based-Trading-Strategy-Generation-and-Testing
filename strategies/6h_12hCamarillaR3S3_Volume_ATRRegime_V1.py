#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation and ATR-based regime filter.
# Long when price breaks above R3 with volume > 1.5x 20-period average and ATR(14) > ATR(50) (expanding volatility).
# Short when price breaks below S3 with same conditions.
# Exit when price returns to the daily pivot point (PP) or ATR contracts below ATR(50).
# Uses discrete position size 0.25. Camarilla levels provide institutional support/resistance,
# volume confirms breakout strength, ATR regime filter avoids false breakouts in choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (R3, S3, PP) ===
    # Camarilla formulas based on previous 12h bar
    pp = (high_12h + low_12h + close_12h) / 3.0
    r3 = pp + (high_12h - low_12h) * 1.1 / 4.0
    s3 = pp - (high_12h - low_12h) * 1.1 / 4.0
    
    # Align 12h Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Get 6h data for volume and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume moving average (20-period) on 6h
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    # True Range for ATR calculation
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50) for regime filter
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_6h, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_6h, atr_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_14_val = atr_14_aligned[i]
        atr_50_val = atr_50_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot point or ATR contracts
            if price <= pp_val or atr_14_val < atr_50_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot point or ATR contracts
            if price >= pp_val or atr_14_val < atr_50_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma_val
            
            # Regime filter: ATR(14) > ATR(50) (expanding volatility)
            regime_filter = atr_14_val > atr_50_val
            
            # LONG: price breaks above R3 with volume and regime confirmation
            if price > r3_val and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below S3 with volume and regime confirmation
            elif price < s3_val and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_12hCamarillaR3S3_Volume_ATRRegime_V1"
timeframe = "6h"
leverage = 1.0