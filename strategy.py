#!/usr/bin/env python3
"""
Experiment #483: 4h Donchian(20) breakout + 12h EMA(21) trend + volume confirmation + chop filter
HYPOTHESIS: 4h Donchian breakouts aligned with 12h EMA(21) trend capture momentum. Volume confirmation (>1.5x average) and choppiness filter (CHOP<38.2 for trending) ensure validity. Discrete sizing (0.25) limits drawdown. Works in bull (breakouts with trend) and bear (shorts on downtrend breakouts). Designed for 75-150 total trades over 4 years to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_483_4h_donchian20_12h_ema21_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA(21) trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = pd.Series(df_12h['close'].values)
    ema_12h = close_12h.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align EMA trend to 4h timeframe (shifted by 1 for completed 12h bar only)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === HTF: 1d data for Choppiness Index regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = high_1d.rolling(window=14, min_periods=14).max().values
    lowest_low_14 = low_1d.rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumATR14 / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    hl_range_14 = highest_high_14 - lowest_low_14
    hl_range_14 = np.where(hl_range_14 == 0, 1e-10, hl_range_14)
    chop_1d = 100 * np.log10(sum_atr_14 / hl_range_14) / np.log10(14)
    
    # Align chop to 4h timeframe (shifted by 1 for completed 1d bar only)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Regime Filter: Only trade in trending markets (CHOP < 38.2) ---
        trending_market = chop_1d_aligned[i] < 38.2
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- 12h EMA Trend Filter ---
        # For long: price above EMA (uptrend)
        # For short: price below EMA (downtrend)
        ema_uptrend = price > ema_12h_aligned[i]
        ema_downtrend = price < ema_12h_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~32h on 4h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike and trending_market:
            # Long: Donchian breakout up + 12h EMA uptrend
            if breakout_up and ema_uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + 12h EMA downtrend
            elif breakout_down and ema_downtrend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals