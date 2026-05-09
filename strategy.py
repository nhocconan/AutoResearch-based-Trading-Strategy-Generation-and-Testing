#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal strategy with daily volume confirmation
# Uses daily Camarilla levels (R3/S3) for mean-reversion entries and R4/S4 for breakout continuation
# Filters trades with 6h volume spike (>1.5x 20-period average) to avoid low-conviction moves
# Designed to work in both bull and bear markets by adapting to volatility regimes
# Target: 15-35 trades/year (60-140 total over 4 years)
name = "6h_Camarilla_Volume_Reversal_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # True range for Camarilla calculation
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - prev_close)
    tr3 = np.abs(prev_low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels
    R4 = prev_close + tr * 1.1 / 2
    R3 = prev_close + tr * 1.1 / 4
    S3 = prev_close - tr * 1.1 / 4
    S4 = prev_close - tr * 1.1 / 2
    
    # Align daily levels to 6h timeframe (wait for daily close)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Mean reversion entry at R3/S3 with volume spike
            if price <= R3_aligned[i] and price >= S3_aligned[i] and vol_spike:
                # Near R3: short bias, near S3: long bias
                mid = (R3_aligned[i] + S3_aligned[i]) / 2
                if price < mid:
                    signals[i] = -0.25  # Short near resistance
                    position = -1
                else:
                    signals[i] = 0.25   # Long near support
                    position = 1
            # Breakout continuation at R4/S4 with volume spike
            elif price > R4_aligned[i] and vol_spike:
                signals[i] = 0.25   # Breakout long
                position = 1
            elif price < S4_aligned[i] and vol_spike:
                signals[i] = -0.25  # Breakdown short
                position = -1
        
        elif position == 1:
            # Exit long: price reaches opposite S3 level or loses volume confirmation
            if price <= S3_aligned[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches opposite R3 level or loses volume confirmation
            if price >= R3_aligned[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals