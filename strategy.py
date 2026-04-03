#!/usr/bin/env python3
"""
Experiment #067: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian breakouts above 20-bar high or below 20-bar low, aligned with 
weekly pivot direction (price above/below weekly pivot from 1w timeframe) and volume 
confirmation (>1.5x average), capture strong momentum moves in both bull and bear markets.
Weekly pivot acts as regime filter: only long when price > weekly pivot, only short when 
price < weekly pivot. This avoids counter-trend trades and improves win rate. 
Target: 75-150 total trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_067_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Weekly Pivot (based on prior week OHLC) ===
    # Calculate weekly pivot using prior week's data to avoid look-ahead
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pivot = np.full_like(c_1w, np.nan)
    for i in range(1, len(c_1w)):
        # Weekly pivot = (Prior week high + low + close) / 3
        weekly_pivot[i] = (h_1w[i-1] + l_1w[i-1] + c_1w[i-1]) / 3.0
    
    # Align to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian and volume stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pivot_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_confirm = vol_ratio[i] > 1.5  # Volume confirmation threshold
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit on Donchian opposite break with volume
            if position_side > 0:  # Long position
                if low[i] < donchian_lower[i-1] and vol_confirm:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if high[i] > donchian_upper[i-1] and vol_confirm:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Only trade in direction of weekly pivot (regime filter)
        if price > weekly_pivot_6h[i]:  # Bullish regime - only look for longs
            if high[i] > donchian_upper[i-1] and vol_confirm:  # Break above upper Donchian
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif price < weekly_pivot_6h[i]:  # Bearish regime - only look for shorts
            if low[i] < donchian_lower[i-1] and vol_confirm:  # Break below lower Donchian
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Price exactly at weekly pivot - no clear regime
            signals[i] = 0.0
    
    return signals