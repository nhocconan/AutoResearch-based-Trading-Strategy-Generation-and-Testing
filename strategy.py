#!/usr/bin/env python3
"""
Experiment #1908: 12h Donchian(20) Breakout + 1w Trend + Volume Confirmation
HYPOTHESIS: 12h Donchian breakouts aligned with 1w trend and volume spikes capture 
institutional momentum moves. Works in both bull/bear by following 1w trend direction.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1908_12h_donchian20_1w_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_50_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 12h Indicators: Donchian(20) channels ===
    # Calculate rolling max/min for 20 periods
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss and profit target ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stops
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                # Simple ATR approximation using recent average
                atr_approx = np.mean([
                    max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                    for j in range(max(1, i-13), i+1)
                ]) if i >= 1 else 0.0
            else:
                atr_approx = 0.0
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5 * ATR below entry
                if price < entry_price - 2.5 * atr_approx:
                    exit_signal = True
                # Take profit: 4 * ATR above entry
                elif price > entry_price + 4.0 * atr_approx:
                    exit_signal = True
                # Time exit: max 10 bars (~5 days)
                elif bars_since_entry >= 10:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2.5 * ATR above entry
                if price > entry_price + 2.5 * atr_approx:
                    exit_signal = True
                # Take profit: 4 * ATR below entry
                elif price < entry_price - 4.0 * atr_approx:
                    exit_signal = True
                # Time exit: max 10 bars
                elif bars_since_entry >= 10:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1w trend alignment
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND 1w trend up
            if trend_bias > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND 1w trend down
            elif trend_bias < 0 and price < low_roll_min[i]:
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