#/usr/bin/env python3
"""
12h_12H_Camarilla_R1_S1_Breakout_1wTrend_VolumeS
Hypothesis: Use weekly Camarilla pivot levels (R1/S1) on 1d for breakout signals, filtered by 1w trend (EMA50) and volume confirmation.
The Camarilla pivot levels act as key support/resistance levels. Breakouts above R1 or below S1 with volume confirmation
and aligned with weekly trend capture strong directional moves. This strategy aims for low trade frequency (12-37/year)
with high win rate by requiring multiple confluence factors.
Target: 50-150 total trades over 4 years on 12h timeframe.
"""

name = "12h_12H_Camarilla_R1_S1_Breakout_1wTrend_VolumeS"
timeframe = "12h"
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
    
    # === 1D Data for Camarilla Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # Using close of previous day to avoid look-ahead
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # Set first day's previous values to NaN (no previous day)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla equations
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === 1W Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-period EMA on weekly close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume Average (20-period on 12h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need at least 50 weeks of data for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, volume confirmation, and price above weekly EMA50
            if close[i] > R1_aligned[i] and vol_confirm and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, volume confirmation, and price below weekly EMA50
            elif close[i] < S1_aligned[i] and vol_confirm and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR weekly trend turns bearish
            if close[i] < S1_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above R1 OR weekly trend turns bullish
            if close[i] > R1_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals