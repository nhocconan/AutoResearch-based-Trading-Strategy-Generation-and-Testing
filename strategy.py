#!/usr/bin/env python3
"""
Experiment: Williams %R Extreme + ATR Expansion + 1d Trend

HYPOTHESIS: Williams %R at extreme levels (<-90 or >-10) marks exhaustion zones.
Combined with:
1. ATR ratio > 1.0 (volatility expansion = institutional move, not noise)
2. 1d EMA21 trend alignment (filters countertrend entries)
3. Volume confirmation (smart money participation)

This catches momentum reversals at extremes while avoiding whipsaws in chop.

WHY 12h: ~3 trades/week = ~150/year max. ATR ratio filter cuts false breakouts.
Target: 50-100 total trades/4 years (tight but valid).

HARD RULES:
- Williams %R must be at extreme (<-90 for long, >-10 for short)
- ATR ratio > 1.0 (vol expanding, not consolidating)
- 1d EMA21 confirms direction
- Volume 1.5x minimum
- 2.5x ATR stoploss
- 3-bar minimum hold to reduce churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_atr_expansion_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_williams_r(high, low, close, period=21):
    """Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    roll_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    roll_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    highest = roll_high.values
    lowest = roll_low.values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        willr = np.where(
            (highest - lowest) > 1e-10,
            -100 * (highest - close) / (highest - lowest),
            -50  # Neutral when range is zero
        )
    
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_21 = calculate_williams_r(high, low, close, period=21)
    
    # ATR ratio: current ATR vs ATR EMA20 (volatility expansion filter)
    atr_ema20 = pd.Series(atr_14).ewm(span=20, min_periods=20, adjust=False).mean().values
    atr_ratio = atr_14 / np.where(atr_ema20 > 0, atr_ema20, 1)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    cooldown = 0
    MIN_HOLD = 3  # Minimum bars to hold (reduces churn)
    
    warmup = 150  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === WILLIAMS %R EXTREME LEVELS ===
        willr = willr_21[i]
        willr_oversold = willr < -90  # Deep oversold
        willr_overbought = willr > -10  # Deep overbought
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ATR EXPANSION (key filter - eliminates consolidation breakouts) ===
        atr_expanding = atr_ratio[i] > 1.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Manage cooldown
        if cooldown > 0:
            cooldown -= 1
        
        # Entry only when cooldown is clear
        if cooldown == 0 and not in_position:
            # === LONG: Oversold + ATR expanding + volume + uptrend ===
            if price_above_1d_ema and willr_oversold and vol_spike and atr_expanding:
                desired_signal = SIZE
            
            # === SHORT: Overbought + ATR expanding + volume + downtrend ===
            if not price_above_1d_ema and willr_overbought and vol_spike and atr_expanding:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            bars_held = i - entry_bar
            
            # Minimum hold period
            if bars_held >= MIN_HOLD:
                # Long exit: %R normalized (>-30) or trend broke
                if position_side > 0:
                    if willr > -30 or not price_above_1d_ema:
                        desired_signal = 0.0
                        cooldown = 3  # Cool down before new entries
                
                # Short exit: %R normalized (<-70) or trend broke
                if position_side < 0:
                    if willr < -70 or price_above_1d_ema:
                        desired_signal = 0.0
                        cooldown = 3
        
        # === POSITION MANAGEMENT ===
        if in_position and np.sign(desired_signal) != position_side:
            # Close current, open new if signal flipped
            in_position = False
            position_side = 0
        
        if desired_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        
        # === STOPLOSS (2.5x ATR) ===
        if in_position and position_side > 0:
            stop = entry_price - 2.5 * entry_atr
            if low[i] < stop:
                desired_signal = 0.0
                in_position = False
                position_side = 0
                cooldown = 3
        
        if in_position and position_side < 0:
            stop = entry_price + 2.5 * entry_atr
            if high[i] > stop:
                desired_signal = 0.0
                in_position = False
                position_side = 0
                cooldown = 3
        
        # === COOLDOWN ENFORCEMENT ===
        if cooldown > 0 and desired_signal != 0.0:
            # Don't enter if in cooldown, maintain 0
            if not in_position:
                desired_signal = 0.0
        
        signals[i] = desired_signal
    
    return signals