#!/usr/bin/env python3
"""
Experiment #235: 6h Elder Ray + Weekly Pivot Regime + Volume Spike

HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure, filtered by weekly pivot regime (above/below weekly pivot = bull/bear bias) and volume spike confirmation. This captures strong momentum moves with institutional participation while avoiding choppy markets. Works in both bull and bear markets by following the prevailing power balance. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA200 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(200) on 1d close
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for weekly pivot points ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Weekly pivot: P = (H+L+C)/3
        weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
        # Weekly R1: 2*P - L
        weekly_r1 = 2 * weekly_pivot - low_1w
        # Weekly S1: 2*P - H
        weekly_s1 = 2 * weekly_pivot - high_1w
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # ATR(14) for volatility and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure (negative values indicate selling pressure)
    
    # Volume confirmation: 1.5x average volume
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
    
    warmup = 200  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Elder Ray Signals ---
        # Strong bullish pressure: Bull Power > 0 and increasing
        bull_power_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        strong_bullish = bull_power[i] > 0 and bull_power_rising
        
        # Strong bearish pressure: Bear Power < 0 and decreasing (more negative)
        bear_power_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        strong_bearish = bear_power[i] < 0 and bear_power_falling
        
        # --- Regime Filters ---
        # Weekly pivot regime: price above/below weekly pivot determines bias
        price_above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # EMA200 trend filter: ensures alignment with higher timeframe trend
        above_ema200 = close[i] > ema_200_1d_aligned[i]
        below_ema200 = close[i] < ema_200_1d_aligned[i]
        
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
            
            # Exit conditions: power weakening or opposite pressure
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: Bull Power turns negative OR Bear Power becomes strong
                    if bull_power[i] <= 0 or strong_bearish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: Bear Power turns positive OR Bull Power becomes strong
                    if bear_power[i] >= 0 or strong_bullish:
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
        # Strong bullish power + price above weekly pivot + above EMA200 + volume confirmation
        if strong_bullish and price_above_weekly_pivot and above_ema200 and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Strong bearish power + price below weekly pivot + below EMA200 + volume confirmation
        elif strong_bearish and price_below_weekly_pivot and below_ema200 and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

}