#!/usr/bin/env python3
"""
Experiment #3239: 6h Williams Alligator + Elder Ray + 12h ADX Regime Filter
HYPOTHESIS: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence on 6h.
Elder Ray (Bull/Bear Power) measures trend strength via EMA13 deviation.
12h ADX > 25 confirms trending regime (avoid whipsaws in ranging markets).
Only trade when Alligator is 'awake' (Lips > Teeth > Jaw for long, reverse for short)
AND Elder Ray confirms direction (Bull Power > 0 for long, Bear Power < 0 for short)
AND 12h ADX > 25. Uses discrete position sizing (0.25) to limit drawdown.
Designed to work in bull markets (trend continuation) and bear markets (trend continuation down)
by requiring strong trending conditions via ADX filter. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3239_6h_alligator_elder_12h_adx_v1"
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
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di) + 1e-10) * 100
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Williams Alligator ===
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
        return pd.Series(arr).ewm(alpha=1/period, adjust=False).mean().values
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition (future leakage prevented by using past values)
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, 13, 8, 5, 13)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when 12h ADX > 25 (trending market) ---
        if adx_12h_aligned[i] <= 25:
            signals[i] = 0.0
            continue
        
        # --- Alligator Awake Conditions ---
        # Long: Lips > Teeth > Jaw (alligator eating up)
        # Short: Lips < Teeth < Jaw (alligator eating down)
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        alligator_long = lips_above_teeth and teeth_above_jaw
        alligator_short = lips_below_teeth and teeth_below_jaw
        
        # --- Elder Ray Confirmation ---
        # Long: Bull Power > 0 (bulls in control)
        # Short: Bear Power < 0 (bears in control)
        elder_long = bull_power[i] > 0
        elder_short = bear_power[i] < 0
        
        # --- Entry Logic ---
        if not in_position:
            if alligator_long and elder_long:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif alligator_short and elder_short:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        # --- Exit Logic ---
        else:
            # Exit conditions: Alligator sleeping or Elder Ray divergence
            alligator_sleeping = not (alligator_long or alligator_short)  # jaws intertwined
            elder_divergence_long = position_side > 0 and bull_power[i] <= 0
            elder_divergence_short = position_side < 0 and bear_power[i] >= 0
            
            if alligator_sleeping or elder_divergence_long or elder_divergence_short:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE if position_side > 0 else -SIZE
    
    return signals