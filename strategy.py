#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week ATR-based breakout and volume confirmation.
# Uses weekly ATR to set dynamic breakout bands from weekly open.
# Long when price breaks above weekly open + 1.5*weekly ATR with volume confirmation.
# Short when price breaks below weekly open - 1.5*weekly ATR with volume confirmation.
# Exits when price returns to weekly open or opposite band is touched.
# Works in both bull and bear markets by capturing breakouts with volatility filter.
# Target: 30-100 total trades over 4 years (7-25/year).
name = "1d_1w_ATRBreakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR and open calculation (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Calculate weekly ATR (14-period)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate breakout bands: weekly open ± 1.5 * weekly ATR
    upper_band = open_1w + 1.5 * atr_1w
    lower_band = open_1w - 1.5 * atr_1w
    
    # Align weekly bands to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    open_1w_aligned = align_htf_to_ltf(prices, df_1w, open_1w)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(open_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout
            if close[i] > upper_band_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout
            elif close[i] < lower_band_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to weekly open or touches lower band
            if close[i] <= open_1w_aligned[i] or close[i] < lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to weekly open or touches upper band
            if close[i] >= open_1w_aligned[i] or close[i] > upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals