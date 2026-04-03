#!/usr/bin/env python3
"""
Experiment #415: 6h Elder Ray + 1d ADX Regime

HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h combined with 1d ADX regime filter captures 
trend strength and momentum in both bull and bear markets. Bull Power > 0 + Bear Power < 0 
with ADX > 25 indicates strong trending conditions. Uses discrete position sizing (0.25) 
to minimize fee drag. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1d_adx_regime_v1"
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
        tr = np.maximum(high_1d - low_1d, 
                        np.maximum(abs(high_1d - np.roll(close_1d, 1)), 
                                   abs(low_1d - np.roll(close_1d, 1))))
        tr[0] = high_1d[0] - low_1d[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                           np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                            np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
        
        # Align to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 20.0)  # Default to ranging market
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    if n >= 13:
        # EMA(13) as the reference
        ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
        
        # Bull Power = High - EMA(13)
        bull_power = high - ema_13
        
        # Bear Power = Low - EMA(13)
        bear_power = low - ema_13
    else:
        ema_13 = np.full(n, np.nan)
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # === Session filter: 00-23 UTC (trade all hours for 6h timeframe) ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Trade all hours for 6h timeframe ---
        hour = hours[i]
        # No session filter for 6h - trade continuously
        
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Regime filter: ADX > 25 indicates trending market
        is_trending = adx_aligned[i] > 25
        
        # Long: Bull Power > 0 (buying pressure) + Bear Power < 0 (weak selling) + trending
        long_condition = (
            bull_power[i] > 0 and      # Buying pressure
            bear_power[i] < 0 and      # Weak selling pressure
            is_trending                # Trending market (ADX > 25)
        )
        
        # Short: Bear Power < 0 (selling pressure) + Bull Power < 0 (weak buying) + trending
        short_condition = (
            bear_power[i] < 0 and      # Selling pressure
            bull_power[i] < 0 and      # Weak buying pressure
            is_trending                # Trending market (ADX > 25)
        )
        
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