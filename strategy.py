#!/usr/bin/env python3
"""
Experiment #271: 6h Elder Ray + 1d ADX Regime Filter

HYPOTHESIS: Elder Ray (Bull Power/Bear Power) identifies institutional buying/selling pressure. 
Combined with 1d ADX regime filter (ADX>25 = trending, ADX<20 = ranging), we take Elder Ray signals 
only in trending regimes. This avoids whipsaws in ranging markets while capturing strong trends. 
Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years). Works in both bull 
and bear markets by only taking signals aligned with the higher timeframe trend regime.
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
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) >= 14:
        # Calculate ADX(14) on daily data
        # True Range
        tr = np.zeros(len(df_1d))
        tr[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
        for i in range(1, len(df_1d)):
            tr[i] = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        
        # Directional Movement
        dm_plus = np.zeros(len(df_1d))
        dm_minus = np.zeros(len(df_1d))
        for i in range(1, len(df_1d)):
            up_move = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
            down_move = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed values
        atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr_14 + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr_14 + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
        
        # Regime: 1 = trending (ADX > 25), 0 = ranging (ADX < 20), -1 = transition
        adx_regime = np.where(adx_aligned > 25, 1, np.where(adx_aligned < 20, 0, -1))
    else:
        adx_aligned = np.full(n, np.nan)
        adx_regime = np.full(n, 0)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # EMA(13) and EMA(26) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_26 = pd.Series(close).ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # Elder Ray Components
    bull_power = high - ema_13  # Bull Power: High - EMA(13)
    bear_power = low - ema_26   # Bear Power: Low - EMA(26)
    
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
        if (np.isnan(atr_14[i]) or np.isnan(ema_13[i]) or np.isnan(ema_26[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) if i < len(adx_aligned) else True or
            i >= len(adx_regime)):
            signals[i] = 0.0
            continue
        
        # --- Elder Ray Signals ---
        # Bullish: Bull Power > 0 AND Bear Power < 0 (strong buying pressure)
        bullish_signal = bull_power[i] > 0 and bear_power[i] < 0
        # Bearish: Bull Power < 0 AND Bear Power > 0 (strong selling pressure)
        bearish_signal = bull_power[i] < 0 and bear_power[i] > 0
        
        # --- Regime Filter: Only trade in trending markets (ADX > 25) ---
        in_trending_regime = adx_regime[i] == 1 if i < len(adx_regime) else False
        
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
            
            # Exit conditions: Elder Ray reversal or regime change to ranging
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: Bear Power turns positive OR regime becomes ranging
                    if bear_power[i] > 0 or adx_regime[i] == 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: Bull Power turns positive OR regime becomes ranging
                    if bull_power[i] > 0 or adx_regime[i] == 0:
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
        # Long conditions: Bullish Elder Ray signal AND trending regime
        if bullish_signal and in_trending_regime:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions: Bearish Elder Ray signal AND trending regime
        elif bearish_signal and in_trending_regime:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals