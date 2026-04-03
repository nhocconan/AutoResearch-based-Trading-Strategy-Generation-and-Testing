#!/usr/bin/env python3
"""
Experiment #162: 12h Donchian(20) Breakout + Daily/Weekly Pivot Trend + Volume Spike

HYPOTHESIS: 12h Donchian breakouts filtered by higher timeframe (1d/1w) pivot trend direction 
and volume spikes capture strong momentum moves with reduced false breakouts. Daily pivot 
provides intermediate trend filter, weekly pivot provides major trend filter. This combination 
works in bull markets (breakouts with volume) and bear markets (failed breaks reverse sharply 
from pivot levels). Target: 50-150 trades over 4 years (12-37/year) with Sharpe > 0 on all symbols.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_162_12h_donchian_daily_weekly_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Daily Pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Daily Pivot (Standard: (H+L+C)/3) and Support/Resistance levels
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    daily_pivot = typical_price_1d.values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Daily R1, S1 (for trend filter: price > daily_pivot = bullish, < daily_pivot = bearish)
    daily_r1 = 2 * daily_pivot - daily_low
    daily_s1 = 2 * daily_pivot - daily_high
    
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # === HTF: 1w data for Weekly Pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Weekly Pivot (Standard: (H+L+C)/3) and Support/Resistance levels
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot = typical_price_1w.values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly R1, S1 (for trend filter: price > weekly_pivot = bullish, < weekly_pivot = bearish)
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF pivot, ATR, and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(daily_pivot_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Higher Timeframe Pivot Trend Filter: 
        # Price > daily_pivot AND price > weekly_pivot = strong bullish bias
        # Price < daily_pivot AND price < weekly_pivot = strong bearish bias
        # Mixed signals = no trade (avoid choppy/transition periods) ---
        price_above_both_pivots = (close[i] > daily_pivot_aligned[i]) and (close[i] > weekly_pivot_aligned[i])
        price_below_both_pivots = (close[i] < daily_pivot_aligned[i]) and (close[i] < weekly_pivot_aligned[i])
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above BOTH daily AND weekly pivot
        long_condition = breakout_up and volume_spike and price_above_both_pivots
        
        # Short: Donchian breakout down + volume spike + price below BOTH daily AND weekly pivot
        short_condition = breakout_down and volume_spike and price_below_both_pivots
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals