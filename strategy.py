#!/usr/bin/env python3
"""
Experiment #171: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with 1d weekly pivot direction (price above/below weekly pivot) and volume confirmation capture institutional order flow. Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Works in bull/bear by only taking breakouts in direction of weekly pivot bias. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_171_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot from prior week (H+L+C)/3
    # Need to resample 1d to weekly - but we can approximate using rolling window
    # Weekly high = max(high) over 5 days, weekly low = min(low) over 5 days, weekly close = close of last day
    # Since we only have daily data, we'll use 5-day rolling for weekly approximation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly approximation: 5-day rolling window
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values  # approx
    
    # Weekly pivot point = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Bias: price above weekly pivot = bullish bias, below = bearish bias
    bias_up = close_1d > weekly_pivot
    bias_down = close_1d < weekly_pivot
    
    # Align to 6h timeframe
    bias_up_aligned = align_htf_to_ltf(prices, df_1d, bias_up)
    bias_down_aligned = align_htf_to_ltf(prices, df_1d, bias_down)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Donchian high = max(high) over 20 periods
    # Donchian low = min(low) over 20 periods
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # enough for Donchian(20) and weekly pivot
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(bias_up_aligned[i]) or np.isnan(bias_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below Donchian low (failed breakout)
                if close[i] < donch_low[i]:
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
                # Exit if price closes above Donchian high (failed breakout)
                if close[i] > donch_high[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Long: Price breaks above Donchian high AND bullish bias AND volume spike
        if high[i] > donch_high[i] and bias_up_aligned[i] and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian low AND bearish bias AND volume spike
        elif low[i] < donch_low[i] and bias_down_aligned[i] and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals