#!/usr/bin/env python3
"""
Experiment #614: 1h Donchian(20) breakout + 4h EMA(50) trend + 1d volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h EMA trend and confirmed by 1d volume spike capture institutional momentum while reducing whipsaw. Using 4h/1d for signal direction and 1h for precise entry timing targets 15-37 trades/year (60-150 over 4 years). Session filter (08-20 UTC) reduces noise. Fixed size 0.20 balances risk and opportunity.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_614_1h_donchian20_4h_ema50_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate volume ratio on 1d: current volume / 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(volume_1d))
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    # Pre-compute session hours filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    warmup = 50  # sufficient for Donchian and EMA calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade during 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require 1d volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- HTF Trend Filter: 4h EMA direction ---
        # Long only when price above 4h EMA (uptrend)
        # Short only when price below 4h EMA (downtrend)
        ema_trend_up = price > ema_4h_aligned[i]
        ema_trend_down = price < ema_4h_aligned[i]
        
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
            
            # Optional: time-based exit after 12 bars (~12h on 1h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + 4h EMA uptrend
            if breakout_up and ema_trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + 4h EMA downtrend
            elif breakout_down and ema_trend_down:
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