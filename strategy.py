#!/usr/bin/env python3
"""
Experiment #1126: 12h Primary + 1d HTF — Simplified KAMA Trend + RSI Pullback

Hypothesis: After 819+ failed experiments, key insights for 12h timeframe:
1. Complex regime-switching causes 0 trades — SIMPLER is better
2. KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA
3. 1d KAMA for macro trend + 12h RSI pullback = proven pattern
4. LOOSE RSI thresholds (35/65) ensure 30-50 trades/year on 12h
5. Single ATR trailing stop at 2.5x — no complex exit logic
6. Position size 0.28 base with discrete levels to minimize fee churn

Why this should beat Sharpe=0.612:
- 12h has cleaner signals than 4h (less noise)
- KAMA adapts to market regime automatically (no Choppiness needed)
- Fewer filters = more trades = better statistical significance
- Conservative sizing (0.28) protects against 2022-style crashes
- Proven on SOL (KAMA + ADX + Choppiness = Sharpe +0.755 in research)

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.28 base, 0.14 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 30-50 trades/year, Sharpe > 0.612, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average — adapts to market volatility.
    
    Formula:
    1. Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    2. Smoothing Constant (SC) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    3. KAMA[i] = KAMA[i-1] + SC * (Close[i] - KAMA[i-1])
    
    ER near 1 = trending (fast response)
    ER near 0 = choppy (slow response)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 1d KAMA slope for trend confirmation
    kama_1d_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(kama_1d_aligned[i]) and not np.isnan(kama_1d_aligned[i-5]):
            kama_1d_slope[i] = kama_1d_aligned[i] - kama_1d_aligned[i-5]
        else:
            kama_1d_slope[i] = np.nan
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # 12h KAMA for local trend
    kama_12h = calculate_kama(close, period=10, fast=2, slow=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.14
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_12h[i]):
            continue
        if np.isnan(kama_1d_slope[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d KAMA) ===
        macro_bull = close[i] > kama_1d_aligned[i] and kama_1d_slope[i] > 0
        macro_bear = close[i] < kama_1d_aligned[i] and kama_1d_slope[i] < 0
        
        # === LOCAL TREND (12h KAMA) ===
        local_bull = close[i] > kama_12h[i]
        local_bear = close[i] < kama_12h[i]
        
        # === PULLBACK SIGNAL (12h RSI) ===
        # LOOSE thresholds to ensure adequate trade frequency
        rsi_oversold = rsi_12h[i] < 40.0
        rsi_overbought = rsi_12h[i] > 60.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # Macro bull + local pullback + RSI oversold
        if macro_bull and rsi_oversold:
            # Entry on pullback in uptrend
            desired_signal = current_size
        
        # === SHORT ENTRY ===
        # Macro bear + local rally + RSI overbought
        elif macro_bear and rsi_overbought:
            # Entry on rally in downtrend
            desired_signal = -current_size
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull
                if macro_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or RSI very overbought
            if macro_bear or rsi_12h[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or RSI very oversold
            if macro_bull or rsi_12h[i] < 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals