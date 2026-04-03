#!/usr/bin/env python3
"""
Experiment #1379: 6h Elder Ray + 12h Regime Filter (ADX)
HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h identifies institutional buying/selling pressure. 
Regime filter from 12h ADX (trending >25, range <20) ensures trades align with market structure. 
In trending markets (ADX>25): enter long when Bear Power crosses above zero (bulls gaining control), 
enter short when Bull Power crosses below zero (bears gaining control). 
In ranging markets (ADX<20): fade extremes - long when Bull Power crosses below zero (overextended longs), 
short when Bear Power crosses above zero (overextended shorts). 
Volume confirmation (>1.5x average) filters for institutional participation. 
Designed to work in both bull (trend following) and bear (mean reversion in ranges) markets. 
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1379_6h_elder_ray_12h_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[high[0] - low[0]], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
        minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    # EMA(13) as proxy for equilibrium
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13 (negative = bearish)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    bars_since_entry = 0
    
    warmup = 20  # sufficient for EMA and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal ---
        if in_position:
            bars_since_entry += 1
            
            # Determine current regime
            adx_val = adx_12h_aligned[i]
            is_trending = adx_val > 25
            is_ranging = adx_val < 20
            
            # Generate exit signal based on regime
            exit_signal = False
            if is_trending:
                # In trending market: exit when power reverses against position
                if position_side > 0:  # Long
                    if bear_power[i] > 0:  # Bears taking control
                        exit_signal = True
                else:  # Short
                    if bull_power[i] < 0:  # Bulls taking control
                        exit_signal = True
            elif is_ranging:
                # In ranging market: exit when power reverses toward mean
                if position_side > 0:  # Long
                    if bull_power[i] < 0:  # Bulls losing strength
                        exit_signal = True
                else:  # Short
                    if bear_power[i] > 0:  # Bears losing strength
                        exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine current regime
            adx_val = adx_12h_aligned[i]
            is_trending = adx_val > 25
            is_ranging = adx_val < 20
            
            if is_trending:
                # Trending market: follow the smart money
                # Enter long when Bear Power crosses above zero (bulls gaining control)
                if bear_power[i] > 0 and bear_power[i-1] <= 0:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Enter short when Bull Power crosses below zero (bears gaining control)
                elif bull_power[i] < 0 and bull_power[i-1] >= 0:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging market: fade the extremes (contrarian)
                # Enter long when Bull Power crosses below zero (overextended longs)
                if bull_power[i] < 0 and bull_power[i-1] >= 0:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Enter short when Bear Power crosses above zero (overextended shorts)
                elif bear_power[i] > 0 and bear_power[i-1] <= 0:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX between 20-25): no trade
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals