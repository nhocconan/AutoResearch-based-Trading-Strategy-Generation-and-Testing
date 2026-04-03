#!/usr/bin/env python3
"""
Experiment #139: 6h Donchian(20) breakout + 12h Supertrend(10,3.0) + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Supertrend direction and volume confirmation (>1.8x) capture medium-term momentum with controlled trade frequency. Supertrend provides adaptive trend filtering that works in both bull and bear markets by dynamically adjusting to volatility. Volume confirmation ensures institutional participation. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_139_6h_donchian20_12h_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend(10,3.0) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for Supertrend
    tr_12h = np.zeros(len(close_12h))
    for i in range(1, len(close_12h)):
        tr_12h[i] = max(high_12h[i] - low_12h[i], 
                        abs(high_12h[i] - close_12h[i-1]), 
                        abs(low_12h[i] - close_12h[i-1]))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Calculate Supertrend
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + (3.0 * atr_12h)
    lower_band_12h = hl2_12h - (3.0 * atr_12h)
    
    upper_band_12h = pd.Series(upper_band_12h).rolling(window=1, min_periods=1).min().values  # only decrease
    lower_band_12h = pd.Series(lower_band_12h).rolling(window=1, min_periods=1).max().values  # only increase
    
    supertrend_12h = np.zeros(len(close_12h))
    supertrend_direction_12h = np.ones(len(close_12h))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upper_band_12h[i-1]:
            supertrend_direction_12h[i] = 1
        elif close_12h[i] < lower_band_12h[i-1]:
            supertrend_direction_12h[i] = -1
        else:
            supertrend_direction_12h[i] = supertrend_direction_12h[i-1]
            
        if supertrend_direction_12h[i] == 1:
            supertrend_12h[i] = lower_band_12h[i]
        else:
            supertrend_12h[i] = upper_band_12h[i]
    
    # Align Supertrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    trend_aligned = align_htf_to_ltf(prices, df_12h, supertrend_direction_12h)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
            np.isnan(vol_ratio[i]) or np.isnan(supertrend_aligned[i]) or
            np.isnan(trend_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- 12h Supertrend Trend ---
        bullish_trend = trend_aligned[i] > 0
        bearish_trend = trend_aligned[i] < 0
        
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
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish 12h Supertrend
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish 12h Supertrend
            elif breakout_down and bearish_trend:
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