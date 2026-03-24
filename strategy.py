#!/usr/bin/env python3
"""
Experiment #1539: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After analyzing #1534 failure (Sharpe=-0.156), the problem is OVER-COMPLEXITY:
1. Dual regime (choppy vs trend) creates conflicting signals
2. Connors RSI thresholds too extreme (<15, >85 rarely fire = 0 trades)
3. Two HTF filters (12h + 1d) can conflict with each other
4. 7-level signal discretization creates fee churn

NEW APPROACH - SIMPLIFIED & PROVEN:
- Single trend-following logic (no regime switching)
- 1d HMA(21) for MACRO bias only (not both 12h and 1d)
- 4h HMA(16/48) crossover for ENTRY timing
- RSI(14) 40-60 for pullback confirmation (NOT extremes - ensures trades fire)
- ATR(14) 2.5x trailing stop
- Signal: discrete 0.0, ±0.30 only (no 0.5, 0.7 multipliers = less fee churn)

Why this should work:
- Research notes: HMA+RSI on 4h gave SOL Sharpe +0.879
- RSI 40-60 fires frequently (unlike CR<15 which rarely triggers)
- Single HTF filter reduces conflicts
- Simpler logic = more consistent execution
- Target: 30-50 trades/year on 4h (within fee drag limits)

Timeframe: 4h (required by experiment)
HTF: 1d (macro trend bias only)
Position Size: 0.30 (conservative for 4h volatility)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades > 30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_trend_rsi_pullback_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
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
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA Crossover) ===
        hma_fast_above_slow = hma_16[i] > hma_48[i]
        hma_fast_below_slow = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK (loose thresholds to ensure trades fire) ===
        # RSI 40-60 = neutral zone, good for pullback entries
        rsi_ok_long = rsi_14[i] >= 40.0 and rsi_14[i] <= 65.0
        rsi_ok_short = rsi_14[i] >= 35.0 and rsi_14[i] <= 60.0
        
        # === HMA SLOPE (trend confirmation) ===
        hma_16_slope = 0.0
        if i >= 5 and not np.isnan(hma_16[i-5]):
            hma_16_slope = (hma_16[i] - hma_16[i-5]) / hma_16[i-5] if hma_16[i-5] > 1e-10 else 0.0
        
        hma_16_rising = hma_16_slope > 0.0
        hma_16_falling = hma_16_slope < 0.0
        
        # === DESIRED SIGNAL — SIMPLIFIED TREND + PULLBACK ===
        desired_signal = 0.0
        
        # LONG: Daily bull + 4h HMA crossover + RSI pullback + HMA rising
        if daily_bull and hma_fast_above_slow and rsi_ok_long and hma_16_rising:
            desired_signal = BASE_SIZE
        
        # SHORT: Daily bear + 4h HMA crossover + RSI pullback + HMA falling
        elif daily_bear and hma_fast_below_slow and rsi_ok_short and hma_16_falling:
            desired_signal = -BASE_SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES (simple: 0, ±0.30 only) ===
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Position flip
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals