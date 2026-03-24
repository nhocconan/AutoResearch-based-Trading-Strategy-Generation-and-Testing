#!/usr/bin/env python3
"""
Experiment #049: 4h Primary + 1d HTF — KAMA Adaptive Trend + Donchian Breakout

Hypothesis: KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA in choppy
markets because it adapts to volatility - moves fast in trends, slow in ranges.
Combined with Donchian breakout confirmation and loose RSI thresholds, this should:
1. Reduce whipsaw in 2022 crash (KAMA slows down in high vol chop)
2. Capture trends when they occur (Donchian breakout confirmation)
3. Generate sufficient trades (loose RSI 40/60 thresholds)
4. Control drawdown (volatility-adjusted position sizing)

Key improvements over #044:
- KAMA instead of HMA (adaptive to market regime)
- Donchian(20) breakout for trend confirmation
- Volatility-adjusted sizing (reduce size when ATR spikes)
- Same loose entry logic to ensure trade generation

Timeframe: 4h (target 20-50 trades/year)
Size: 0.25-0.30 base, reduced to 0.15-0.20 in high vol
Target: Beat Sharpe=0.313, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_rsi_loose_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - fast in trends, slow in chop
    From Perry Kaufman's "Trading Systems and Methods"
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i - period):i + 1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[period] = close[period]  # Initialize with price
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    # Set early values to NaN
    kama[:period] = np.nan
    
    return kama

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds for trade generation"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss and volatility adjustment"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - for breakout confirmation"""
    n = len(close) if (close := high) else len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for HTF trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_4h_fast = calculate_kama(close, period=5, fast_period=2, slow_period=15)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate ATR ratio for volatility adjustment
    atr_short = calculate_atr(high, low, close, period=7)
    atr_ratio = np.zeros(n)
    for i in range(14, n):
        if atr[i] > 1e-10:
            atr_ratio[i] = atr_short[i] / atr[i] if not np.isnan(atr_short[i]) else 1.0
        else:
            atr_ratio[i] = 1.0
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size
    
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
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 4h TREND (KAMA) ===
        trend_bull = close[i] > kama_4h[i]
        trend_bear = close[i] < kama_4h[i]
        kama_fast_above_slow = kama_4h_fast[i] > kama_4h[i] if not np.isnan(kama_4h_fast[i]) else False
        kama_fast_below_slow = kama_4h_fast[i] < kama_4h[i] if not np.isnan(kama_4h_fast[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.995
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.005
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        # Reduce size when volatility spikes (ATR ratio > 1.5)
        if atr_ratio[i] > 1.5:
            size_multiplier = 0.6  # Reduce to 60% in high vol
        elif atr_ratio[i] > 1.2:
            size_multiplier = 0.8  # Reduce to 80% in medium vol
        else:
            size_multiplier = 1.0  # Full size in normal vol
        
        current_size = BASE_SIZE * size_multiplier
        
        # === DESIRED SIGNAL (LOOSE thresholds for trade generation) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + (4h trend bull OR Donchian breakout OR RSI pullback)
        # Multiple entry conditions to ensure trade generation
        if htf_bull:
            if trend_bull and kama_fast_above_slow:
                # Strong uptrend - enter
                desired_signal = current_size
            elif donchian_breakout_long and rsi[i] < 65.0:
                # Breakout confirmation with RSI filter
                desired_signal = current_size
            elif rsi[i] < 45.0 and trend_bull:
                # Pullback in uptrend - buy the dip (loose threshold)
                desired_signal = current_size
        
        # SHORT: HTF bear + (4h trend bear OR Donchian breakdown OR RSI rally)
        # Multiple entry conditions to ensure trade generation
        if htf_bear:
            if trend_bear and kama_fast_below_slow:
                # Strong downtrend - enter
                desired_signal = -current_size
            elif donchian_breakout_short and rsi[i] > 35.0:
                # Breakdown confirmation with RSI filter
                desired_signal = -current_size
            elif rsi[i] > 55.0 and trend_bear:
                # Rally in downtrend - sell the rip (loose threshold)
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        # Use discrete levels to minimize fee churn
        if desired_signal >= current_size * 0.85:
            final_signal = current_size
        elif desired_signal <= -current_size * 0.85:
            final_signal = -current_size
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