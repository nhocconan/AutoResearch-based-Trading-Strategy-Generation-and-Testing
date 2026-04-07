#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + Volume Spike + Choppiness Regime
# Hypothesis: Camarilla pivot levels (from daily) act as strong support/resistance.
# Enter long at S1/S2 on bounce with volume spike in choppy market (mean reversion).
# Enter short at R1/R2 on rejection with volume spike in choppy market.
# Choppiness filter avoids trending markets where reversals fail.
# Designed for 4h timeframe with low trade frequency (20-50/year).
# Works in bull via mean reversion at support, in bear via mean reversion at resistance.

name = "4h_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Formula: 
    # H4 = C + ((H-L) * 1.1/2)
    # H3 = C + ((H-L) * 1.1/4)
    # H2 = C + ((H-L) * 1.1/6)
    # H1 = C + ((H-L) * 1.1/12)
    # L1 = C - ((H-L) * 1.1/12)
    # L2 = C - ((H-L) * 1.1/6)
    # L3 = C - ((H-L) * 1.1/4)
    # L4 = C - ((H-L) * 1.1/2)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    H4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    H3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    H2 = close_1d + ((high_1d - low_1d) * 1.1 / 6)
    H1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    L1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    L2 = close_1d - ((high_1d - low_1d) * 1.1 / 6)
    L3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    L4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align levels to 4h time (use previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Choppiness Index (14-period)
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range14 = highest_high14 - lowest_low14
    range14[range14 == 0] = 1e-10
    
    chop = 100 * np.log10(atr14 * 14 / range14) / np.log10(14)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(chop[i]) or np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: only trade in choppy markets (61.8 < chop < 100)
        chop_ok = chop[i] > 61.8
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H1 (take profit) or closes below L2 (stop)
            if close[i] >= H1_aligned[i] or close[i] <= L2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches L1 (take profit) or closes above H2 (stop)
            if close[i] <= L1_aligned[i] or close[i] >= H2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if chop_ok and vol_ok:
                # Long: price touches or goes below L2 and closes back above it (bounce)
                if low[i] <= L2_aligned[i] and close[i] > L2_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches or goes above H2 and closes back below it (rejection)
                elif high[i] >= H2_aligned[i] and close[i] < H2_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals