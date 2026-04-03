#!/usr/bin/env python3
"""
Experiment #078: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
HYPOTHESIS: Price breaking 1d Donchian(20) channels with alignment to 1w EMA50 trend and 1d volume spike captures swing momentum with proper weekly trend context. Weekly EMA50 provides institutional trend filter that works in both bull (price above EMA50) and bear (price below EMA50) markets. Daily volume confirmation ensures institutional participation. Discrete sizing (0.25) and ATR(14) stoploss (2.0*ATR). Target: 40-80 total trades over 4 years (10-20/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_078_1d_donchian20_1w_ema50_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA50 trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_1d_ratio = np.ones(len(df_1d))  # default to 1.0 for warmup period
    vol_1d_ratio[20:] = df_1d['volume'].iloc[20:].values / vol_ma_1d[20:]
    vol_1d_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_ratio)
    
    # === 1d Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_1d_ratio_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- 1w EMA50 Trend Filter ---
        trend_up = price > ema_1w_aligned[i]
        trend_down = price < ema_1w_aligned[i]
        
        # --- 1d Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_1d_ratio_aligned[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
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
            
            # Optional: time-based exit after 25 bars (~25 days) to avoid overtrading
            if bars_since_entry > 25:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND above 1w EMA50 (bullish bias)
            if breakout_up and trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND below 1w EMA50 (bearish bias)
            elif breakout_down and trend_down:
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