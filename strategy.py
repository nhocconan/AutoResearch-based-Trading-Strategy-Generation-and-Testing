#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Reversal with 1d Volume Filter
# Uses daily Camarilla pivot levels calculated from prior day's OHLC.
# Enters long at S1 support or short at R1 resistance with volume confirmation (>1.5x avg volume).
# Exit when price touches opposite pivot level (S2/R2) or reverses at same level.
# Works in both bull/bear by fading extremes at key intraday support/resistance.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla multipliers
    # S1 = C - (H-L)*1.12/2
    # S2 = C - (H-L)*1.12/4
    # R1 = C + (H-L)*1.12/2
    # R2 = C + (H-L)*1.12/4
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.12 / 2
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.12 / 4
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.12 / 2
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.12 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for daily candle to close)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    
    # Volume confirmation: volume > 1.5x average volume (24-period = 2 days)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 25
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price touches S1 support with volume confirmation
            if abs(price - s1_aligned[i]) < 0.001 * price and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price touches R1 resistance with volume confirmation
            elif abs(price - r1_aligned[i]) < 0.001 * price and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches S2 (further support) or reverses at S1
            if price <= s2_aligned[i] or abs(price - s1_aligned[i]) < 0.0005 * price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches R2 (further resistance) or reverses at R1
            if price >= r2_aligned[i] or abs(price - r1_aligned[i]) < 0.0005 * price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_Volume_Filter"
timeframe = "12h"
leverage = 1.0