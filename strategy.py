#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation
    # Works in both bull and bear: Camarilla levels provide structured support/resistance,
    # volume confirms institutional interest, discrete sizing minimizes fee drag.
    # Target: 20-40 trades/year to stay within 4h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L4 = C - Range * 1.1/2
    # L3 = C - Range * 1.1/4
    # L2 = C - Range * 1.1/6
    # L1 = C - Range * 1.1/12
    # H1 = C + Range * 1.1/12
    # H2 = C + Range * 1.1/6
    # H3 = C + Range * 1.1/4
    # H4 = C + Range * 1.1/2
    
    # Use previous day's data (shifted by 1)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    
    # Camarilla levels
    L4 = prev_close - rng * 1.1 / 2
    L3 = prev_close - rng * 1.1 / 4
    L2 = prev_close - rng * 1.1 / 6
    L1 = prev_close - rng * 1.1 / 12
    H1 = prev_close + rng * 1.1 / 12
    H2 = prev_close + rng * 1.1 / 6
    H3 = prev_close + rng * 1.1 / 4
    H4 = prev_close + rng * 1.1 / 2
    
    # Get 4h volume for confirmation (20-period average)
    vol_avg_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, prices, vol_avg_20_4h)  # self-align
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(L4_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(vol_avg_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_4h_aligned[i]
        
        # Entry conditions: 
        # Long: price crosses above L3 with volume (bullish bounce from support)
        # Short: price crosses below H3 with volume (bearish rejection from resistance)
        enter_long = (close[i] > L3_aligned[i] and close[i-1] <= L3_aligned[i-1]) and volume_confirmed
        enter_short = (close[i] < H3_aligned[i] and close[i-1] >= H3_aligned[i-1]) and volume_confirmed
        
        # Exit conditions: 
        # Long exit: price reaches L1 (strong support) or H3 (resistance)
        # Short exit: price reaches H1 (strong resistance) or L3 (support)
        exit_long = position == 1 and (close[i] <= L1_aligned[i] or close[i] >= H3_aligned[i])
        exit_short = position == -1 and (close[i] >= H1_aligned[i] or close[i] <= L3_aligned[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0