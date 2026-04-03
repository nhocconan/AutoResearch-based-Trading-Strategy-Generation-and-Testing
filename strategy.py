#!/usr/bin/env python3
"""
Experiment #075: 6h Weekly Pivot Reversal with Volume Spike and ATR Filter

HYPOTHESIS: Price reversals at weekly pivot levels (R1/S1, R2/S2) with volume confirmation
capture institutional order flow exhaustion. Weekly pivots act as magnet levels where
smart money takes profit or initiates new positions. Volume spike (>1.5x average) confirms
participation. ATR filter ensures sufficient momentum. Designed for 15-25 trades/year
to minimize fee drag while maintaining statistical significance. Works in both bull/bear
markets by fading extremes and trading continuations at higher pivot levels.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_weekly_pivot(df_daily):
    """Calculate weekly pivot levels from daily OHLC data."""
    n = len(df_daily)
    if n < 5:
        return {
            'PP': np.full(n, np.nan),
            'R1': np.full(n, np.nan), 'S1': np.full(n, np.nan),
            'R2': np.full(n, np.nan), 'S2': np.full(n, np.nan),
            'R3': np.full(n, np.nan), 'S3': np.full(n, np.nan)
        }
    
    high = pd.Series(df_daily['high'].values)
    low = pd.Series(df_daily['low'].values)
    close = pd.Series(df_daily['close'].values)
    
    weekly_high = high.rolling(window=5, min_periods=5).max()
    weekly_low = low.rolling(window=5, min_periods=5).min()
    weekly_close = close.rolling(window=5, min_periods=5).last()
    
    PP = (weekly_high + weekly_low + weekly_close) / 3.0
    R1 = 2 * PP - weekly_low
    S1 = 2 * PP - weekly_high
    R2 = PP + (weekly_high - weekly_low)
    S2 = PP - (weekly_high - weekly_low)
    R3 = weekly_high + 2 * (PP - weekly_low)
    S3 = weekly_low - 2 * (weekly_high - PP)
    
    return {
        'PP': PP.values,
        'R1': R1.values, 'S1': S1.values,
        'R2': R2.values, 'S2': S2.values,
        'R3': R3.values, 'S3': S3.values
    }

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    pivot_data = calculate_weekly_pivot(df_1d)
    
    # Align all pivot levels to LTF
    PP_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['PP'])
    R1_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['R1'])
    S1_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['S1'])
    R2_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['R2'])
    S2_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['S2'])
    R3_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['R3'])
    S3_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['S3'])
    
    # === 6h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Profit taking: reduce position at 2R profit
            if position_side > 0:
                profit_level = entry_price + 2.0 * atr_14[i]
                if high[i] >= profit_level and signals[i] == SIZE:
                    signals[i] = SIZE * 0.5  # Reduce to half position
            else:  # Short position
                profit_level = entry_price - 2.0 * atr_14[i]
                if low[i] <= profit_level and signals[i] == -SIZE:
                    signals[i] = -SIZE * 0.5  # Reduce to half position
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE if abs(signals[i]) == SIZE else signals[i]
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Calculate distances to pivot levels for proximity check
        dist_to_R1 = abs(close[i] - R1_aligned[i]) / close[i] if close[i] > 0 else float('inf')
        dist_to_S1 = abs(close[i] - S1_aligned[i]) / close[i] if close[i] > 0 else float('inf')
        dist_to_R2 = abs(close[i] - R2_aligned[i]) / close[i] if close[i] > 0 else float('inf')
        dist_to_S2 = abs(close[i] - S2_aligned[i]) / close[i] if close[i] > 0 else float('inf')
        dist_to_R3 = abs(close[i] - R3_aligned[i]) / close[i] if close[i] > 0 else float('inf')
        dist_to_S3 = abs(close[i] - S3_aligned[i]) / close[i] if close[i] > 0 else float('inf')
        
        # Proximity threshold: within 0.3% of pivot level
        proximity_threshold = 0.003
        
        near_R1 = dist_to_R1 < proximity_threshold
        near_S1 = dist_to_S1 < proximity_threshold
        near_R2 = dist_to_R2 < proximity_threshold
        near_S2 = dist_to_S2 < proximity_threshold
        near_R3 = dist_to_R3 < proximity_threshold
        near_S3 = dist_to_S3 < proximity_threshold
        
        # Momentum check: price must be moving away from pivot after touch
        if i >= 2:
            price_momentum = (close[i] - close[i-2]) / close[i-2] if close[i-2] > 0 else 0
        else:
            price_momentum = 0
        
        # Long entries:
        # 1. Near S1/S2/S3 with bullish momentum (bounce)
        # 2. Near R3 with bearish momentum (breakout continuation)
        if (near_S1 or near_S2 or near_S3) and price_momentum > 0.001 and vol_ok:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            signals[i] = SIZE
        elif near_R3 and price_momentum > 0.002 and vol_ok:  # Breakout continuation
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short entries:
        # 1. Near R1/R2/R3 with bearish momentum (rejection)
        # 2. Near S3 with bullish momentum (breakdown continuation)
        elif (near_R1 or near_R2 or near_R3) and price_momentum < -0.001 and vol_ok:
            in_position = True
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        elif near_S3 and price_momentum < -0.002 and vol_ok:  # Breakdown continuation
            in_position = True
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals