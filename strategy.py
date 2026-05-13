#!/usr/bin/env python3
# 6h_PivotBreakout_VolumeRegime
# Hypothesis: Buy breakouts above R3 and sell breakdowns below S3 of daily Camarilla pivots,
# filtered by 12h trend and volume spike. Works in both bull and bear markets by
# capturing momentum bursts during regime shifts while avoiding chop.
# Uses daily pivots for structure, 12h EMA for trend, and volume spike for confirmation.
# Targets 15-30 trades/year with disciplined entries to minimize fee drag.

name = "6h_PivotBreakout_VolumeRegime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily OHLC
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # But we only need R3 and S3 for entry, R4/S4 for stop (not implemented here)
    # Actually, standard Camarilla: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # We use R3 and S3 as entry triggers
    
    # Calculate once for the whole daily array
    H_1d = df_1d['high'].values
    L_1d = df_1d['low'].values
    C_1d = df_1d['close'].values
    
    # Avoid division by zero and handle NaN
    diff = H_1d - L_1d
    # Only calculate where we have valid data
    valid = ~(np.isnan(H_1d) | np.isnan(L_1d) | np.isnan(C_1d) | (diff == 0))
    
    R3 = np.full_like(C_1d, np.nan)
    S3 = np.full_like(C_1d, np.nan)
    R3[valid] = C_1d[valid] + (diff[valid] * 1.1 / 4)
    S3[valid] = C_1d[valid] - (diff[valid] * 1.1 / 4)
    
    # Get 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # EMA50 on 12h for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detector: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Align all to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))  # bool to float for alignment
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Minimum bars needed for indicators
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume spike confirmation (current bar)
        vol_spike = volume_spike_aligned[i] > 0.5  # Convert back to boolean
        
        # Trend filter: 12h EMA50
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # Breakout conditions
        # Long: price breaks above R3 with volume spike in uptrend
        long_breakout = (close[i] > R3_aligned[i]) and vol_spike and uptrend
        # Short: price breaks below S3 with volume spike in downtrend
        short_breakout = (close[i] < S3_aligned[i]) and vol_spike and downtrend
        
        # Exit conditions: opposite breakout or loss of trend
        exit_long = (close[i] < S3_aligned[i]) or (not uptrend)
        exit_short = (close[i] > R3_aligned[i]) or (not downtrend)
        
        if position == 0:
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals