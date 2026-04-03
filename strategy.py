#!/usr/bin/env python3
"""
Experiment #027: 6h Elder Ray Power + 1d ADX Regime Filter

HYPOTHESIS: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) 
measures bull/bear strength relative to trend. Combined with 1d ADX regime filter 
(ADX > 25 = trending market), this captures strong directional moves while 
avoiding choppy markets. The 6h timeframe provides sufficient signal quality 
to minimize fee drag, targeting 50-150 total trades over 4 years (12-37/year). 
Works in both bull (buy strength) and bear (sell weakness) markets by fading 
extreme power readings when trend is confirmed.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 20.0)  # Default to ranging market
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    if n >= 13:
        ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    else:
        ema_13 = close.copy()
    
    # Elder Ray Power
    bull_power = high - ema_13   # Bull Power = High - EMA
    bear_power = low - ema_13    # Bear Power = Low - EMA
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for EMA13 and HTF ADX
    
    for i in range(warmup, n):
        # Skip if ADX data not ready
        if np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 25 (trending market) ---
        is_trending = adx_aligned[i] > 25.0
        
        if not is_trending:
            # In ranging markets, stay flat to avoid whipsaw
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit when bull power weakens (fading strength)
                if bull_power[i] < 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit when bear power weakens (fading weakness)
                if bear_power[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Strong bull power (price significantly above EMA) in uptrend
        long_condition = bull_power[i] > 0 and bull_power[i] > np.percentile(bull_power[max(0, i-50):i+1], 80)
        
        # Short: Strong bear power (price significantly below EMA) in downtrend
        short_condition = bear_power[i] < 0 and bear_power[i] < np.percentile(bear_power[max(0, i-50):i+1], 20)
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals