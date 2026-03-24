#!/usr/bin/env python3
"""
Experiment #106: 12h Primary + 1d HTF — KAMA Trend + Fisher Transform Reversals

Hypothesis: After 100+ experiments, clear patterns emerge:
- 12h timeframe is underutilized but proven (SOL +0.782 with Donchian+HMA)
- Fisher Transform excels in bear/range markets (2025 test period)
- KAMA adapts better than HMA/EMA to volatility changes
- Simple is better: complex regime filters cause 0 trades (#096, #105 failed)
- Loose entry + strict exit = more trades with controlled drawdown

This strategy combines:
1. 1d KAMA = major trend bias (price above/below)
2. 12h Fisher Transform = reversal timing (crosses -1.5/+1.5)
3. 12h KAMA crossover = trend confirmation
4. ATR volatility filter = avoid extreme vol entries (ATR ratio < 2.5)
5. Trailing stoploss (2.5x ATR) = risk management

Key design choices:
- Timeframe: 12h (20-50 trades/year target, proven higher TF works)
- HTF: 1d for trend bias (responsive but not noisy)
- Fisher Transform: period=9, catches reversals in bear rallies
- KAMA: adaptive smoothing, no separate regime filter needed
- Position size: 0.30 (30% of capital, conservative for 12h)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_reversal_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    """
    n = len(close)
    if n < period + slow + 5:
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
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA with SMA of first period
    kama[period] = np.mean(close[:period+1])
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into a Gaussian normal distribution
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(high)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize (avoid division by zero)
        if hh - ll < 1e-10:
            continue
        
        # Normalized price (-1 to +1)
        norm = 2.0 * (typical - ll) / (hh - ll) - 1.0
        
        # Clamp to avoid extreme values
        norm = max(-0.999, min(0.999, norm))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + norm) / (1.0 - norm))
        
        # Previous fisher (for crossover detection)
        if i > period:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss and volatility filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(atr, short_period=7, long_period=30):
    """ATR ratio for volatility spike detection"""
    n = len(atr)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    atr_short = pd.Series(atr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    ratio = np.zeros(n)
    ratio[:] = np.nan
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

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
    
    # Calculate primary (12h) indicators
    kama_fast = calculate_kama(close, period=10, fast=2, slow=15)
    kama_slow = calculate_kama(close, period=30, fast=2, slow=30)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr, short_period=7, long_period=30)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 12h)
    
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
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]) or atr_ratio[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 12h TREND (KAMA crossover) ===
        kama_cross_bull = kama_fast[i] > kama_slow[i]
        kama_cross_bear = kama_fast[i] < kama_slow[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === VOLATILITY FILTER ===
        # Avoid entries during extreme volatility spikes (ATR ratio > 2.5)
        vol_ok = atr_ratio[i] < 2.5
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 12h KAMA bull + Fisher long + vol ok
        # SHORT: 1d bear + 12h KAMA bear + Fisher short + vol ok
        desired_signal = 0.0
        
        if htf_bull and kama_cross_bull and fisher_long and vol_ok:
            desired_signal = SIZE
        elif htf_bear and kama_cross_bear and fisher_short and vol_ok:
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