#!/usr/bin/env python3
"""
Experiment #191: 6h Elder Ray + 1d ADX Trend Filter
HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h identifies momentum exhaustion, while 1d ADX > 25 confirms trending regime. Long when Bear Power turns up in uptrend (ADX>25, +DI>-DI), short when Bull Power turns down in downtrend. Uses ATR stoploss (2.0x) and discrete sizing (0.25). Target: 75-150 total trades over 4 years. Works in bull via Bear Power mean reversion in uptrend, and in bear via Bull Power mean reversion in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_191_6h_elder_ray_1d_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX and DI (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX and DI on 1d
    def calculate_adx_di(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        return adx, plus_di, minus_di
    
    adx_1d, plus_di_1d, minus_di_1d = calculate_adx_di(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    # === 6h Indicators: Elder Ray (Bull Power, Bear Power) ===
    # EMA(13) as proxy for equilibrium
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA
    bear_power = low - ema_13   # Bear Power: Low - EMA
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(plus_di_1d_aligned[i]) or np.isnan(minus_di_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Trend Condition: 1d ADX > 25 and DI alignment ---
        is_uptrend = adx_1d_aligned[i] > 25 and plus_di_1d_aligned[i] > minus_di_1d_aligned[i]
        is_downtrend = adx_1d_aligned[i] > 25 and minus_di_1d_aligned[i] > plus_di_1d_aligned[i]
        
        # --- Elder Ray Signals ---
        # Bull Power turning up (from negative to less negative or positive)
        bull_power_up = bull_power[i] > bull_power[i-1]
        # Bear Power turning down (from positive to less positive or negative)
        bear_power_down = bear_power[i] < bear_power[i-1]
        
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
                # Exit long if Bear Power turns down in uptrend
                if is_uptrend and bear_power_down:
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
                # Exit short if Bull Power turns up in downtrend
                if is_downtrend and bull_power_up:
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
        # Long: Bear Power turning up in uptrend (mean reversion long)
        if is_uptrend and bear_power_down == False and bear_power[i] < 0 and bull_power_up:
            # Additional confirmation: Bear Power improving (less negative)
            if bear_power[i] > bear_power[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
        # Short: Bull Power turning down in downtrend (mean reversion short)
        elif is_downtrend and bull_power_up == False and bull_power[i] > 0 and bear_power_down:
            # Additional confirmation: Bull Power worsening (less positive)
            if bull_power[i] < bull_power[i-1]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals