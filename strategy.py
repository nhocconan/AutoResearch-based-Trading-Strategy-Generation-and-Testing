#!/usr/bin/env python3
"""
Experiment #5091: 6h Elder Ray + ADX Regime + Volume Confirmation (Refined)
HYPOTHESIS: On 6h timeframe, Elder Ray (Bull Power/Bear Power) identifies bullish/bearish momentum, 
while ADX(14) > 25 filters for trending regimes. Volume > 1.5x average confirms participation. 
This combination avoids ranging markets and captures strong trends in both bull (long when Bull Power > 0) 
and bear (short when Bear Power < 0) regimes. Designed for 12-37 trades/year on 6h timeframe 
to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5091_6h_elder_ray_adx_regime_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 6h Indicators: EMA(13) for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray Components ===
    bull_power = high - ema13  # Bull Power = High - EMA(13)
    bear_power = low - ema13   # Bear Power = Low - EMA(13)
    
    # === 6h Indicators: ADX(14) for trend strength ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # ADX
    dx = np.abs(di_plus - di_minus) / (np.abs(di_plus) + np.abs(di_minus)) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(13, 14, 20)  # EMA13, ADX, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposing signal or stoploss ---
        if in_position:
            # Stoploss: 2.5 * ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Exit on opposing Elder Ray signal
            long_exit = bear_power[i] >= 0  # Bear Power turns positive
            short_exit = bull_power[i] <= 0  # Bull Power turns negative
            
            if (position_side > 0 and long_exit) or (position_side < 0 and short_exit):
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: ADX > 25 (trending regime)
        trending = adx[i] > 25
        
        # Entry conditions
        long_entry = (bull_power[i] > 0) and trending and vol_confirm
        short_entry = (bear_power[i] < 0) and trending and vol_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals