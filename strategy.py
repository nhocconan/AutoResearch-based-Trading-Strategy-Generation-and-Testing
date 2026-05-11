#!/usr/bin/env python3
"""
6h_1d_Pivot_Zone_Fade_With_Volume
Hypothesis: Fade price rejection at daily pivot zones (R1/S1, R2/S2) with volume confirmation.
- Long when: price rejects below S1 (drops then closes back above S1) with volume > 20-period average
- Short when: price rejects above R1 (rises then closes back below R1) with volume > 20-period average
- Uses 1d trend filter (EMA50) to avoid counter-trend trades in strong trends
- Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag
- Works in both bull/bear: mean reversion in ranges, avoids strong trends via filter
"""

name = "6h_1d_Pivot_Zone_Fade_With_Volume"
timeframe = "6h"
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
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Daily Pivot Levels (Standard) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots from previous day's data
    pivot_high = np.full_like(close_1d, np.nan)
    pivot_low = np.full_like(close_1d, np.nan)
    pivot_close = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's pivots
        pivot_high[i] = high_1d[i-1]
        pivot_low[i] = low_1d[i-1]
        pivot_close[i] = close_1d[i-1]
    
    # Standard pivot point calculation
    PP = (pivot_high + pivot_low + pivot_close) / 3.0
    R1 = 2 * PP - pivot_low
    S1 = 2 * PP - pivot_high
    R2 = PP + (pivot_high - pivot_low)
    S2 = PP - (pivot_high - pivot_low)
    
    # Align pivot levels to 6h timeframe
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    
    # --- Volume Confirmation: 6h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(R2_6h[i]) or np.isnan(S2_6h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend (avoid counter-trend trades in strong trends)
        trend_up = close_6h[i] > ema50_1d_aligned[i]
        trend_down = close_6h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        # Rejection signals: price moves to level then closes back inside
        if position == 0:
            # Long setup: price tested S1 support and bounced
            # Condition: low touched or went below S1, but close is back above S1
            tested_S1 = low_6h[i] <= S1_6h[i]
            closed_above_S1 = close_6h[i] > S1_6h[i]
            rejection_long = tested_S1 and closed_above_S1
            
            # Short setup: price tested R1 resistance and rejected
            # Condition: high touched or went above R1, but close is back below R1
            tested_R1 = high_6h[i] >= R1_6h[i]
            closed_below_R1 = close_6h[i] < R1_6h[i]
            rejection_short = tested_R1 and closed_below_R1
            
            # Only take rejection trades in direction of 1d trend (or if ranging)
            # In strong uptrend: only take long rejections at S1
            # In strong downtrend: only take short rejections at R1
            # In ranging (no clear trend): take both
            if rejection_long and vol_ok:
                if trend_up or not (trend_up or trend_down):  # uptrend or ranging
                    signals[i] = 0.25
                    position = 1
            elif rejection_short and vol_ok:
                if trend_down or not (trend_up or trend_down):  # downtrend or ranging
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions: return to opposite S2/R2 level or trend reversal
            if position == 1:
                # Exit long: price reaches S2 support or trend turns down
                if close_6h[i] <= S2_6h[i] or (not trend_up and close_6h[i] < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches R2 resistance or trend turns up
                if close_6h[i] >= R2_6h[i] or (not trend_down and close_6h[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals