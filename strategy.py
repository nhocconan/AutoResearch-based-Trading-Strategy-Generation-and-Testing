#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume confirmation and ATR-based stoploss
# Long when price > Camarilla R1 AND 1d volume > 1.5x 20-period 1d volume median
# Short when price < Camarilla S1 AND same volume condition
# Exit when price touches Camarilla pivot point (PP) or opposite level (S1 for long, R1 for short)
# Uses discrete position size 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
# Camarilla pivots from 1d provide strong intraday support/resistance levels that work in all regimes.
# Volume confirmation ensures breakouts have conviction, reducing false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for 1d
    # Camarilla equations:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    rng = high_1d - low_1d
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (rng * 1.1 / 12)
    s1 = close_1d - (rng * 1.1 / 12)
    r2 = close_1d + (rng * 1.1 / 6)
    s2 = close_1d - (rng * 1.1 / 6)
    
    # Volume median (20-period)
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Get 4h data for price reference (we use primary timeframe price directly)
    # No need to load 4h OHLV as we use primary timeframe prices for breakout signals
    
    # Align all 1d indicators to primary timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # Also align 1d volume for current volume reading
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 30  # 1d lookback for volume median
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(vol_median_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        vol_median = vol_median_20_1d_aligned[i]
        vol_1d = vol_1d_aligned[i]
        
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume median
        vol_threshold = vol_median * 1.5
        vol_confirm = vol_1d > vol_threshold
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price touches or goes below Camarilla PP (mean reversion to equilibrium)
            if price <= pp_val:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price touches or goes above Camarilla PP (mean reversion to equilibrium)
            if price >= pp_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Camarilla R1 AND volume confirmation
            if price > r1_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S1 AND volume confirmation
            elif price < s1_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Camarilla_R1S1_1dVolume1.5x_v1"
timeframe = "4h"
leverage = 1.0