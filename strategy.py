#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Squeeze
# Hypothesis: Uses Camarilla pivot levels (S1/S2/R1/R2) from daily data with Bollinger Band squeeze
# (low volatility breakout) to capture breakouts in both bull and bear markets. 
# Entry when price breaks S2/R1 with volume confirmation and Bollinger Band width < 50th percentile.
# Exit when price returns to S1/R2 or Bollinger Band width > 80th percentile (volatility expansion).
# Designed for 4h timeframe with 1d/1w HTF context to limit trades (target: 20-50/year).

name = "4h_Camarilla_Pivot_Squeeze"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Band width regime filter (20, 2)
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_percentile = bb_width.rolling(window=50, min_periods=20).rank(pct=True) * 100
    bb_width_squeeze = bb_width_percentile < 50  # Low volatility regime
    bb_width_expansion = bb_width_percentile > 80  # High volatility exit
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L) * 1.5/2), R3 = C + ((H-L) * 1.25/2), R2 = C + ((H-L) * 1.1/2), R1 = C + ((H-L) * 1.05/2)
    # S1 = C - ((H-L) * 1.05/2), S2 = C - ((H-L) * 1.1/2), S3 = C - ((H-L) * 1.25/2), S4 = C - ((H-L) * 1.5/2)
    # We use S2, S1, R1, R2 for entries/exits
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's values (shifted by 1 to avoid look-ahead)
    range_1d = high_1d - low_1d
    # Use previous day's data for today's levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Calculate Camarilla levels
    R1 = prev_close_1d + (range_1d * 1.05 / 2)
    S1 = prev_close_1d - (range_1d * 1.05 / 2)
    R2 = prev_close_1d + (range_1d * 1.1 / 2)
    S2 = prev_close_1d - (range_1d * 1.1 / 2)
    
    # Align levels to 4h timeframe (using previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(bb_width_squeeze[i]) or np.isnan(bb_width_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume squeeze and volume spike
            if (close[i] > R1_aligned[i] and 
                bb_width_squeeze[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S2 with volume squeeze and volume spike
            elif (close[i] < S2_aligned[i] and 
                  bb_width_squeeze[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to S2 OR volatility expansion
            if (close[i] < S2_aligned[i]) or bb_width_expansion[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to R1 OR volatility expansion
            if (close[i] > R1_aligned[i]) or bb_width_expansion[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals