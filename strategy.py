#!/usr/bin/env python3
"""
Experiment #300: 6h Elder Ray Index + 1d ADX Regime + Volume Confirmation

HYPOTHESIS: Elder Ray (Bull/Bear Power) measures bull/bear strength relative to EMA13. 
Combined with 1d ADX regime filter (ADX>25 = trending, ADX<20 = ranging) and volume 
confirmation (1.5x average), this strategy captures strong momentum in trending markets 
while avoiding false signals in ranging markets. Designed for 6h timeframe targeting 
12-37 trades/year (50-150 over 4 years). Works in both bull/bear by only taking signals 
in direction of higher timeframe trend and power.
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
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) >= 30:
        # Calculate ADX(14) on daily data
        # True Range
        tr = np.maximum(df_1d['high'].values - df_1d['low'].values,
                        np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                   np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
        tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
        
        # Directional Movement
        up_move = df_1d['high'].values - np.roll(df_1d['high'].values, 1)
        down_move = np.roll(df_1d['low'].values, 1) - df_1d['low'].values
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                result[period-1] = np.mean(data[:period])
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr_1d = wilder_smooth(tr, 14)
        plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
        minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = wilder_smooth(dx_1d, 14)
        
        # Align ADX to 6h timeframe
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
        
        # Regime: 1 = trending (ADX>25), 0 = ranging (ADX<20), -1 = transitional
        adx_regime = np.where(adx_1d_aligned > 25, 1, 
                             np.where(adx_1d_aligned < 20, 0, -1))
    else:
        adx_1d_aligned = np.full(n, np.nan)
        adx_regime = np.full(n, -1)  # default to transitional
    
    # === 6h Indicators ===
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR(14) for stoploss
    tr_6h = np.maximum(high - low,
                       np.maximum(np.abs(high - np.roll(close, 1)),
                                  np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(adx_1d_aligned[i]) if i < len(adx_1d_aligned) else True or
            i >= len(adx_regime)):
            signals[i] = 0.0
            continue
        
        # --- Elder Ray Signals ---
        # Strong bull power: bullish momentum
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.abs(bear_power[i])
        # Strong bear power: bearish momentum
        strong_bear = bear_power[i] < 0 and np.abs(bear_power[i]) > bull_power[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume
        
        # --- Regime Filter: Only trade in trending markets (ADX>25) ---
        regime_ok = adx_regime[i] == 1  # Only trending regime
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: power reversal or regime change to ranging
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: bear power becomes stronger OR regime turns ranging
                    if bear_power[i] > 0 or adx_regime[i] == 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: bull power becomes stronger OR regime turns ranging
                    if bull_power[i] < 0 or adx_regime[i] == 0:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Strong bull power + volume confirmation + trending regime
        if strong_bull and vol_ok and regime_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Strong bear power + volume confirmation + trending regime
        elif strong_bear and vol_ok and regime_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals