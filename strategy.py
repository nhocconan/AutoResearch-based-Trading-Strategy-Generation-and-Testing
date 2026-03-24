#!/usr/bin/env python3
"""
Experiment #1047: 6h Primary + 1d HTF — Donchian Breakout + Volume + HTF Trend Filter

Hypothesis: 6h timeframe is ideal for breakout strategies - fast enough to capture
multi-day moves, slow enough to filter noise. Donchian channel breakouts with volume
confirmation and 1d trend filter should outperform mean-reversion strategies on 6h.

Key innovations:
1. Donchian(20) breakout: 20-bar high/low for entry triggers (Turtle Trading classic)
2. Volume confirmation: volume > 1.5x 20-bar average (filters false breakouts)
3. 1d HMA(21) trend filter: only trade breakouts in HTF trend direction
4. ROC(10) momentum filter: confirms breakout has momentum (ROC > 2% for long)
5. ATR(14) 2.5x stoploss with trailing
6. Time-based exit: exit after 10 bars if no profit (prevents dead capital)
7. Discrete sizing: 0.0, ±0.25, ±0.30

Why 6h Donchian should work:
- 6h captures 3-5 day moves (Donchian 20 = ~5 days)
- Volume filter avoids fake breakouts (common in crypto)
- 1d HTF filter prevents counter-trend trades (major failure mode)
- ROC momentum ensures breakout has follow-through
- Time exit prevents capital tied in stalled positions

Entry conditions (LOOSE to guarantee trades):
- LONG: price > Donchian_high(20) + volume > 1.5x avg + close > 1d_HMA + ROC(10) > 1%
- SHORT: price < Donchian_low(20) + volume > 1.5x avg + close < 1d_HMA + ROC(10) < -1%

Exit conditions:
- Stoploss: 2.5x ATR from entry (trailing for longs, inverse for shorts)
- Time exit: 10 bars with no profit > 0.5R
- Signal flip: opposite direction signal

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_vol_htf_trend_1d_v1"
timeframe = "6h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def calculate_volume_ma(volume, period=20):
    """Volume moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    roc_10 = calculate_roc(close, period=10)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss and time exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    max_profit_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(roc_10[i]) or np.isnan(vol_ma_20[i]):
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
        
        # === HTF TREND FILTER (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Long breakout: price crosses above Donchian upper
        long_breakout = close[i] > donchian_upper[i]
        # Short breakout: price crosses below Donchian lower
        short_breakout = close[i] < donchian_lower[i]
        
        # === MOMENTUM FILTER (ROC) ===
        momentum_long = roc_10[i] > 1.0  # >1% momentum
        momentum_short = roc_10[i] < -1.0  # <-1% momentum
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Donchian breakout + volume + HTF bull + momentum
        if long_breakout and volume_confirmed and hma_1d_bull and momentum_long:
            desired_signal = SIZE_STRONG
        # SHORT: Donchian breakout + volume + HTF bear + momentum
        elif short_breakout and volume_confirmed and hma_1d_bear and momentum_short:
            desired_signal = -SIZE_STRONG
        # Weaker signals without volume confirmation
        elif long_breakout and hma_1d_bull and momentum_long:
            desired_signal = SIZE_BASE
        elif short_breakout and hma_1d_bear and momentum_short:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (10 bars with no profit) ===
        time_exit_triggered = False
        if in_position:
            bars_held = i - entry_bar
            if bars_held >= 10 and max_profit_since_entry < 0.5 * entry_atr:
                time_exit_triggered = True
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
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                max_profit_since_entry = 0.0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            # Track max profit for time exit
            if position_side > 0:
                current_profit = high[i] - entry_price
            else:
                current_profit = entry_price - low[i]
            max_profit_since_entry = max(max_profit_since_entry, current_profit)
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                entry_bar = 0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                max_profit_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals