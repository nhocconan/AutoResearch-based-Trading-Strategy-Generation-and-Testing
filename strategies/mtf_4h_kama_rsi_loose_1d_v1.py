#!/usr/bin/env python3
"""
Experiment #101: 4h Primary + 1d HTF — KAMA Adaptive Trend + Loose RSI Filter

Hypothesis: After analyzing 100+ failed experiments, the pattern is clear:
- Complex regime filters (Choppiness, dual-regime) cause 0 trades or negative Sharpe
- KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA
- 4h timeframe with 1d HTF bias has shown success (SOL +0.879 with HMA)
- VERY loose RSI thresholds (25/75) ensure trade generation on all symbols

This strategy uses MINIMAL but effective filters:
1. 1d KAMA = major trend bias (price above/below)
2. 4h KAMA crossover (adaptive periods based on ER) = entry trigger
3. RSI loose filter (>25 for long, <75 for short) - ensures trades generate
4. ATR trailing stoploss (2.5x) for risk management
5. NO Choppiness, NO complex regime detection

Key design choices:
- Timeframe: 4h (proven to work, 20-50 trades/year target)
- HTF: 1d for trend bias (more responsive than 1w)
- KAMA: adapts to market volatility automatically (no need for separate regime filter)
- RSI thresholds: 25/75 (very loose, ensures trades on BTC/ETH/SOL)
- Position size: 0.28 (28% of capital, conservative for 4h)
- Stoploss: 2.5x ATR trailing (tighter than 3x for better risk control)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_loose_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    ER = |price change| / sum(|price changes|) over period
    High ER = trending (fast SC), Low ER = choppy (slow SC)
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close[period:] - close[:-period])
    sum_price_change = np.zeros(n - period)
    for i in range(n - period):
        sum_price_change[i] = np.sum(np.abs(np.diff(close[i:i+period+1])))
    
    # Avoid division by zero
    er = np.zeros(n)
    for i in range(period, n):
        if sum_price_change[i-period] > 1e-10:
            er[i] = price_change[i-period] / sum_price_change[i-period]
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    # SC = ER * (fast_sc - slow_sc) + slow_sc
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA with SMA of first period
    kama[period] = np.mean(close[:period+1])
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    
    # Calculate and align 1d KAMA for major trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=20, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_fast = calculate_kama(close, period=10, fast=2, slow=15)
    kama_slow = calculate_kama(close, period=30, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 4h)
    
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
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        # Simple: is price above or below daily KAMA?
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 4h TREND (KAMA crossover) ===
        kama_cross_bull = kama_fast[i] > kama_slow[i]
        kama_cross_bear = kama_fast[i] < kama_slow[i]
        
        # === RSI FILTER (VERY LOOSE - ensure trades generate on all symbols) ===
        # For longs: RSI > 25 (not extremely oversold, allows more entries)
        # For shorts: RSI < 75 (not extremely overbought, allows more entries)
        rsi_ok_long = rsi[i] > 25.0
        rsi_ok_short = rsi[i] < 75.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h KAMA cross bull + RSI > 25
        # SHORT: 1d bear + 4h KAMA cross bear + RSI < 75
        desired_signal = 0.0
        
        if htf_bull and kama_cross_bull and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and kama_cross_bear and rsi_ok_short:
            desired_signal = -SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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
                # Flip position
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