#!/usr/bin/env python3
"""
Experiment #072: 12h Donchian Breakout + Volume + ADX Regime Filter
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume confirmation (>1.5x average)
and ADX>25 indicate strong momentum continuation. In ranging markets (ADX<20), fade at
Donchian channel extremes. This adapts to bull/bear regimes via ADX filtering. Uses 1d HTF
for trend filter (price above/below 50 EMA). Target: 75-150 trades over 4 years on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_072_12h_donchian_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (50 EMA) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 12h Indicators: ADX(14) for regime detection ===
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for ADX, Donchian, volume stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        
        # --- Regime Detection ---
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions based on regime
            if is_trending:
                # In trending market: exit on opposite Donchian break with volume
                if position_side > 0:  # Long
                    if low[i] < donchian_lower[i-1] and vol_spike:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if high[i] > donchian_upper[i-1] and vol_spike:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Ranging market
                # Exit when price returns to middle of channel
                if position_side > 0:  # Long
                    if price < (donchian_upper[i-1] + donchian_lower[i-1]) / 2:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if price > (donchian_upper[i-1] + donchian_lower[i-1]) / 2:
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
        if is_trending:
            # Trending market: breakout continuation with HTF filter
            # Long: price above Donchian upper + above HTF EMA50
            # Short: price below Donchian lower + below HTF EMA50
            if price > donchian_upper[i-1] and price > ema_50_1d_aligned[i] and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donchian_lower[i-1] and price < ema_50_1d_aligned[i] and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif is_ranging:
            # Ranging market: fade at Donchian extremes
            if price < donchian_upper[i-1] and price > (donchian_upper[i-1] + donchian_lower[i-1]) * 0.8 and vol_spike:
                # Near upper band from below - short fade
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif price > donchian_lower[i-1] and price < (donchian_upper[i-1] + donchian_lower[i-1]) * 0.2 and vol_spike:
                # Near lower band from above - long fade
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            # Transition regime (ADX between 20-25) - no trade
            signals[i] = 0.0
    
    return signals