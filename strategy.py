#!/usr/bin/env python3
"""
Experiment #4875: 6h Camarilla Pivot Reversal with Weekly Trend Filter
HYPOTHESIS: On 6h timeframe, price reversals at Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by weekly trend (price above/below weekly EMA20) capture high-probability swings. 
Uses volume confirmation (>1.5x average) to avoid false signals. Designed for 12-37 trades/year 
on 6h timeframe to minimize fee drag while maintaining statistical significance. 
Works in bull markets (buy R3/S3 bounce in uptrend) and bear markets (sell R3/S3 rejection in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4875_6h_camarilla_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === Weekly Indicators: EMA20 for trend filter ===
    if len(df_1w) >= 20:
        close_1w = df_1w['close'].values
        ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    else:
        ema_1w = np.full(len(df_1w), np.nan)
    
    # Align HTF EMA20 to 6h timeframe
    if len(ema_1w) > 0:
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Need daily OHLC for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate Camarilla levels for each day
        camarilla_r3 = np.full(len(df_1d), np.nan)
        camarilla_s3 = np.full(len(df_1d), np.nan)
        camarilla_r4 = np.full(len(df_1d), np.nan)
        camarilla_s4 = np.full(len(df_1d), np.nan)
        camarilla_r3l = np.full(len(df_1d), np.nan)
        camarilla_s3l = np.full(len(df_1d), np.nan)
        camarilla_r4l = np.full(len(df_1d), np.nan)
        camarilla_s4l = np.full(len(df_1d), np.nan)
        
        for i in range(len(df_1d)):
            # Camarilla pivot formula (based on previous day)
            if i > 0:
                ph = high_1d[i-1]  # previous high
                pl = low_1d[i-1]   # previous low
                pc = close_1d[i-1] # previous close
                
                pivot = (ph + pl + pc) / 3
                range_ = ph - pl
                
                camarilla_r3[i] = pc + range_ * 1.1 / 2
                camarilla_s3[i] = pc - range_ * 1.1 / 2
                camarilla_r4[i] = pc + range_ * 1.1
                camarilla_s4[i] = pc - range_ * 1.1
        
        # Align daily Camarilla levels to 6h timeframe (use previous day's levels)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
        
        # For breakout signals, we need the same day's levels (not shifted)
        camarilla_r3_same = camarilla_r3_aligned  # already aligned with shift(1) from align_htf_to_ltf
        camarilla_s3_same = camarilla_s3_aligned
        camarilla_r4_same = camarilla_r4_aligned
        camarilla_s4_same = camarilla_s4_aligned
    else:
        camarilla_r3_aligned = camarilla_s3_aligned = camarilla_r4_aligned = camarilla_s4_aligned = np.full(n, np.nan)
        camarilla_r3_same = camarilla_s3_same = camarilla_r4_same = camarilla_s4_same = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20)  # Volume MA, need at least 1 day for pivots
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or stoploss ---
        if in_position:
            # Exit conditions: opposite signal or price moves against position
            if position_side > 0:  # Long
                # Exit if price reaches R4 (breakout continuation) or reverses from R3
                if price >= camarilla_r4_same[i] or (price <= camarilla_s3_aligned[i] and camarilla_s3_aligned[i] > 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price reaches S4 (breakdown continuation) or reverses from S3
                if price <= camarilla_s4_same[i] or (price >= camarilla_r3_aligned[i] and camarilla_r3_aligned[i] > 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Weekly trend filter
        weekly_uptrend = price > ema_1w_aligned[i]
        weekly_downtrend = price < ema_1w_aligned[i]
        
        # Mean reversion at R3/S3 levels
        long_setup = (price <= camarilla_s3_aligned[i] * 1.002) and (price >= camarilla_s3_aligned[i] * 0.998) and weekly_uptrend and vol_confirm
        short_setup = (price >= camarilla_r3_aligned[i] * 0.998) and (price <= camarilla_r3_aligned[i] * 1.002) and weekly_downtrend and vol_confirm
        
        # Breakout continuation at R4/S4 levels
        breakout_long = (price >= camarilla_r4_same[i] * 0.998) and weekly_uptrend and vol_confirm
        breakout_short = (price <= camarilla_s4_same[i] * 1.002) and weekly_downtrend and vol_confirm
        
        # Final entry conditions
        if long_setup or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_setup or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals