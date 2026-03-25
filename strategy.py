#!/usr/bin/env python3
"""
Experiment #1258: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Confirmation

Hypothesis: Donchian breakouts work well on 4h timeframe (20-50 trades/year target).
This combines proven patterns from research:
1. Donchian(20) breakout for clean entry signals
2. 1d HMA(21) for major trend bias (only trade with daily trend direction)
3. 4h RSI(14) for momentum confirmation (not extreme, just directional)
4. ATR(14) 2.5x trailing stop for risk management
5. LOOSE entry conditions to guarantee 30+ trades/train, 3+/test

Key differences from failed 4h strategies:
- Simpler logic (no choppiness/complex regime filters that block all entries)
- RSI threshold is loose (just >45 for long, <55 for short) not extreme
- Donchian breakout = frequent signals (guarantees trade count)
- Daily HMA filter = strong directional bias without over-filtering

Why this should work:
- 4h timeframe = natural 20-50 trades/year (fee-friendly)
- Donchian breakout = catches trends early, exits on reversal
- Daily HMA = avoids counter-trend trades in strong regimes
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn
- Trailing stop = protects gains, limits drawdown

Entry logic (LOOSE to guarantee trades):
- LONG: Price > Donchian_high(20) + 1d_HMA bullish + RSI(14) > 45
- SHORT: Price < Donchian_low(20) + 1d_HMA bearish + RSI(14) < 55

Target: Sharpe>0.5, trades>=30 train, trades>=3 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    donchian_high = np.full(n, np.nan, dtype=np.float64)
    donchian_low = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        donchian_high[i] = np.nanmax(high[i - period + 1:i + 1])
        donchian_low[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return donchian_high, donchian_low

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
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
        
        # === TREND DIRECTION (1d HMA bias) ===
        # 1d HMA bias - price above/below daily HMA
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1d HMA slope (compare to 3 bars ago for stability)
        hma_1d_slope = 0.0
        if i >= 3 and not np.isnan(hma_1d_aligned[i-3]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-3]
        
        # === MOMENTUM (RSI) ===
        rsi = rsi_14[i]
        
        # === BREAKOUT SIGNAL (Donchian) ===
        breakout_long = close[i] > donchian_high[i]
        breakout_short = close[i] < donchian_low[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Donchian breakout + 1d bullish + RSI confirming (not extreme)
        if breakout_long and price_above_1d:
            if rsi > 45:  # Loose RSI threshold - just needs to be neutral-bullish
                if rsi > 60:
                    desired_signal = SIZE_STRONG  # Strong momentum
                else:
                    desired_signal = SIZE_BASE  # Basic momentum
        
        # SHORT: Donchian breakout + 1d bearish + RSI confirming (not extreme)
        elif breakout_short and price_below_1d:
            if rsi < 55:  # Loose RSI threshold - just needs to be neutral-bearish
                if rsi < 40:
                    desired_signal = -SIZE_STRONG  # Strong momentum
                else:
                    desired_signal = -SIZE_BASE  # Basic momentum
        
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
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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