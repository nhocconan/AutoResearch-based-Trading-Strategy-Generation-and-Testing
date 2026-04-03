#!/usr/bin/env python3
"""
Experiment #131: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (calculated from prior week's range) 
capture institutional order flow around key psychological levels. Weekly pivot acts as dynamic S/R: 
price breaking above weekly R1 with volume continues upward; breaking below S1 continues downward. 
Volume confirmation ensures participation. Works in bull/bear markets by trading breakouts in direction 
of weekly pivot bias. Targets 15-35 trades/year on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_wma(data, window):
    """Weighted Moving Average"""
    if len(data) < window:
        return np.full(len(data), np.nan)
    weights = np.arange(1, window + 1, dtype=np.float64)
    return np.convolve(data, weights[::-1], mode='valid') / weights.sum()

def calculate_hma(close, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(data, window):
        if len(data) < window:
            return np.full(len(data), np.nan)
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(data, weights[::-1], mode='valid') / weights.sum()
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(half) - WMA(full)
    diff = 2 * np.concatenate([np.full(half - 1, np.nan), wma_half]) - np.concatenate([np.full(period - 1, np.nan), wma_full])
    
    # WMA of diff with sqrt_period
    hma = wma(diff, sqrt_period)
    # Adjust for padding
    hma = np.concatenate([np.full(sqrt_period - 1, np.nan), hma])
    
    return hma

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # === Calculate Weekly Pivot Points from Daily Data ===
    # Group daily data into weeks (starting Monday)
    # We'll calculate weekly pivot for each week and align to 6h bars
    n_1d = len(df_1d_close)
    
    # Arrays to store weekly pivot levels for each daily bar
    weekly_pivot = np.full(n_1d, np.nan)
    weekly_r1 = np.full(n_1d, np.nan)
    weekly_s1 = np.full(n_1d, np.nan)
    weekly_r2 = np.full(n_1d, np.nan)
    weekly_s2 = np.full(n_1d, np.nan)
    
    # Calculate weekly OHLC from daily data
    week_start_idx = 0
    for i in range(n_1d):
        # Check if this is the last day of the week (Friday) or end of data
        # Assuming data starts on arbitrary day, we'll use 5-day weeks
        if i >= 4 and (i - week_start_idx) >= 4:  # 5 days in week
            # Weekly OHLC: Monday's open to Friday's close
            week_high = np.max(df_1d_high[week_start_idx:i+1])
            week_low = np.min(df_1d_low[week_start_idx:i+1])
            week_close = df_1d_close[i]  # Friday's close
            
            # Calculate pivot points
            pp = (week_high + week_low + week_close) / 3.0
            r1 = 2 * pp - week_low
            s1 = 2 * pp - week_high
            r2 = pp + (week_high - week_low)
            s2 = pp - (week_high - week_low)
            
            # Fill the entire week with these values
            for j in range(week_start_idx, i+1):
                weekly_pivot[j] = pp
                weekly_r1[j] = r1
                weekly_s1[j] = s1
                weekly_r2[j] = r2
                weekly_s2[j] = s2
            
            week_start_idx = i + 1
        elif i == n_1d - 1:  # Handle last incomplete week
            if week_start_idx < n_1d:
                week_high = np.max(df_1d_high[week_start_idx:n_1d])
                week_low = np.min(df_1d_low[week_start_idx:n_1d])
                week_close = df_1d_close[n_1d-1]
                
                pp = (week_high + week_low + week_close) / 3.0
                r1 = 2 * pp - week_low
                s1 = 2 * pp - week_high
                r2 = pp + (week_high - week_low)
                s2 = pp - (week_high - week_low)
                
                for j in range(week_start_idx, n_1d):
                    weekly_pivot[j] = pp
                    weekly_r1[j] = r1
                    weekly_s1[j] = s1
                    weekly_r2[j] = r2
                    weekly_s2[j] = s2
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # === 6h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Bias ---
        # Price above weekly R1 = bullish bias
        # Price below weekly S1 = bearish bias
        # Between R1 and S1 = neutral (wait for breakout)
        price_vs_r1 = close[i] - weekly_r1_aligned[i]
        price_vs_s1 = close[i] - weekly_s1_aligned[i]
        
        bullish_bias = price_vs_r1 > 0  # Above R1
        bearish_bias = price_vs_s1 < 0  # Below S1
        neutral = (price_vs_r1 <= 0) and (price_vs_s1 >= 0)  # Between S1 and R1
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
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
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below weekly S1
                    if close[i] <= dc_lower_20[i] or close[i] < weekly_s1_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above weekly R1
                    if close[i] >= dc_upper_20[i] or close[i] > weekly_r1_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with bullish weekly bias and volume confirmation
        if bullish_breakout and bullish_bias and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish weekly bias and volume confirmation
        elif bearish_breakout and bearish_bias and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals