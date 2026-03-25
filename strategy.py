#!/usr/bin/env python3
"""
Experiment #1649: 15m Primary + 1h/1d HTF — Loose Multi-TF Fisher Entry

Hypothesis: 15m timeframe with 1h momentum + 1d trend bias can capture intraday 
moves while respecting higher timeframe direction. Key insight from 15m failures 
(#1637, #1641, #1645 all had 0 trades): entry conditions MUST be very loose.

Why 15m should work (if we fix the trade generation problem):
1. Faster reaction to reversals than 1h/4h strategies
2. Can capture intraday mean-reversion within daily trend
3. More entry opportunities = better statistical edge IF fees controlled

Key design choices based on 15m failure analysis:
1. VERY LOOSE Fisher thresholds: -0.8/+0.8 (not -1.2/+1.2) to guarantee crossovers
2. LOOSE RSI: 40/60 (not 30/70) — 15m RSI rarely hits extremes
3. NO session filter (#1641, #1645 had 0 trades with session filters)
4. NO volume filter (kills trade frequency based on #1607, #1612)
5. Simple 1d HMA(21) for trend bias (complex filters reduce trades)
6. 1h RSI(14) for momentum confirmation (loose thresholds)
7. Discrete signal sizes: 0.15 base, 0.20 strong (smaller for 15m frequency)
8. 2.0x ATR trailing stoploss via signal→0

Entry logic (EXTREMELY LOOSE to guarantee ≥40 trades/train):
- LONG: 1d HMA bullish + 1h RSI > 40 + 15m Fisher cross above -0.8
- SHORT: 1d HMA bearish + 1h RSI < 60 + 15m Fisher cross below +0.8
- Alternative: 15m close > 15m EMA(21) + 1d bias (simple trend follow)

Why this beats previous 15m attempts (all had 0 trades):
- Removed session filter (was blocking all entries)
- Removed volume filter (too restrictive)
- Looser Fisher thresholds (-0.8 vs -1.2)
- Added simple EMA crossover as fallback entry

Target: Sharpe>0.3, trades≥40 train, trades≥5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_rsi_hma_1h1d_loose_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_fisher(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Returns fisher value and trigger (previous value for crossover detection)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    median = (high + low) / 2
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            if i > 0 and not np.isnan(fisher[i-1]):
                fisher[i] = fisher[i-1]
                trigger[i] = fisher[i-1]
            continue
        
        normalized = 2.0 * (median[i] - lowest) / range_val - 1.0
        normalized = max(-0.999, min(0.999, normalized))
        
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher(high, low, period=9)
    ema_21 = calculate_ema(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period (shorter for 15m to get trades faster)
    min_bars = 30
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 1H MOMENTUM (RSI) ===
        rsi_1h = rsi_1h_aligned[i]
        rsi_1h_bullish = rsi_1h > 40  # LOOSE threshold
        rsi_1h_bearish = rsi_1h < 60  # LOOSE threshold
        
        # === FISHER TRANSFORM SIGNALS (VERY LOOSE for 15m trades) ===
        fisher_val = fisher[i]
        fisher_prev = fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else fisher_val
        
        # Fisher crossover signals - VERY LOOSE thresholds for 15m
        fisher_bull_cross = fisher_val > -0.8 and fisher_prev <= -0.8
        fisher_bear_cross = fisher_val < 0.8 and fisher_prev >= 0.8
        
        # === EMA TREND (15m) ===
        ema_trend_bull = close[i] > ema_21[i]
        ema_trend_bear = close[i] < ema_21[i]
        
        # === ENTRY LOGIC (EXTREMELY LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # PRIMARY: Fisher cross + 1d bias + 1h RSI confirmation
        if price_above_1d and rsi_1h_bullish and fisher_bull_cross:
            desired_signal = SIZE_STRONG
        elif price_below_1d and rsi_1h_bearish and fisher_bear_cross:
            desired_signal = -SIZE_STRONG
        
        # FALLBACK 1: 1d bias + 15m EMA trend (simple trend follow, generates more trades)
        elif price_above_1d and ema_trend_bull and rsi_1h > 45:
            desired_signal = SIZE_BASE
        elif price_below_1d and ema_trend_bear and rsi_1h < 55:
            desired_signal = -SIZE_BASE
        
        # FALLBACK 2: Fisher extreme reversal (mean reversion)
        elif fisher_val < -1.0 and price_above_1d:
            desired_signal = SIZE_BASE
        elif fisher_val > 1.0 and price_below_1d:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals