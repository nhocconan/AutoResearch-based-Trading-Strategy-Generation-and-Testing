#!/usr/bin/env python3
"""
Experiment #653: 4h Donchian(20) breakout + 12h EMA alignment + volume confirmation + chop filter
HYPOTHESIS: 4h Donchian breakouts aligned with 12h EMA(34) trend, confirmed by volume spike and low choppiness (trending regime), capture strong directional moves with controlled overtrading. Uses ATR-based stoploss and time-based exit to limit drawdown. Target: 75-200 total trades over 4 years (19-50/year) with Sharpe > 0.5 on test.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_653_4h_donchian20_12h_ema_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend and chop filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA(34) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h choppiness index (CHOP) for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        tr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        tr[0] = high_arr[0] - low_arr[0]
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        chop = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            if atr[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(sum(tr[i-period+1:i+1]) / 
                                        (np.log10(period) * (highest_high[i] - lowest_low[i]))) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop_12h = calculate_chop(high_12h, low_12h, close_12h, period=14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for 12h EMA(34) and Donchian calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(chop_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.7x average) ---
        volume_spike = vol_ratio[i] > 1.7
        
        # --- Chop Filter: Trending regime (CHOP < 40) ---
        trending_regime = chop_12h_aligned[i] < 40.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- EMA Trend Filter ---
        # In uptrend: price > EMA(34)
        # In downtrend: price < EMA(34)
        uptrend = price > ema_12h_aligned[i]
        downtrend = price < ema_12h_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~24h on 4h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike and trending_regime:
            # Long: Donchian breakout up + uptrend
            if breakout_up and uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + downtrend
            elif breakout_down and downtrend:
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