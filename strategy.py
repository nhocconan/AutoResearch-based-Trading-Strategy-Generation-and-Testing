#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Touch_Reversal
Hypothesis: Mean reversion at extreme Camarilla levels (R4/S4) with 1d trend filter.
- Long when: price touches or crosses S4 AND 1d EMA50 is rising (bullish bias)
- Short when: price touches or crosses R4 AND 1d EMA50 is falling (bearish bias)
- Exit when price returns to the 1d close (pivot point) or opposite extreme
Uses 1d trend to avoid counter-trend trades. Targets 15-30 trades/year.
"""

name = "4h_1d_Camarilla_Pivot_Touch_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 1d Close for Exit ---
    close_1d_arr = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_arr)
    
    # --- Camarilla Pivots from 1d (previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots from previous day's OHLC
    camarilla_high = np.full_like(close_1d, np.nan)
    camarilla_low = np.full_like(close_1d, np.nan)
    camarilla_close = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        camarilla_high[i] = high_1d[i-1]
        camarilla_low[i] = low_1d[i-1]
        camarilla_close[i] = close_1d[i-1]
    
    # Calculate Camarilla levels
    H_L = camarilla_high - camarilla_low
    C = camarilla_close
    
    R4 = C + (H_L * 1.5000)
    R3 = C + (H_L * 1.2500)
    R2 = C + (H_L * 1.1666)
    R1 = C + (H_L * 1.0833)
    PP = C
    S1 = C - (H_L * 1.0833)
    S2 = C - (H_L * 1.1666)
    S3 = C - (H_L * 1.2500)
    S4 = C - (H_L * 1.5000)
    
    # Align pivots to 4h timeframe
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    PP_4h = align_htf_to_ltf(prices, df_1d, PP)
    
    # Volume filter: avoid low-volume false signals
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R4_4h[i]) or np.isnan(S4_4h[i]) or 
            np.isnan(PP_4h[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend (using price vs EMA50)
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation: avoid low-volume noise
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for reversals at extreme levels WITH trend alignment
            if low_4h[i] <= S4_4h[i] and trend_up and vol_ok:
                # Long: price touches S4 (extreme support) in 1d uptrend
                signals[i] = 0.25
                position = 1
            elif high_4h[i] >= R4_4h[i] and trend_down and vol_ok:
                # Short: price touches R4 (extreme resistance) in 1d downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to 1d close (pivot point)
                if close_4h[i] >= PP_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to 1d close (pivot point)
                if close_4h[i] <= PP_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals