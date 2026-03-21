#!/usr/bin/env python3
"""
EXPERIMENT #039 - HMA Trend + RSI Pullback + Z-Score Filter + ATR Trailing Stop
====================================================================================
Hypothesis: Building on #038's framework but fixing the indentation crash and adding
a Z-score filter to avoid entering when price is already extended from the mean.
This should reduce false entries at trend exhaustion points.

Key improvements over #038:
- Fixed all indentation errors (was crashing at line 316)
- Added Z-score(20) filter - only enter when |z| < 1.5 (not overextended)
- Simplified position tracking logic to avoid bugs
- Position size capped at 0.28 max (more conservative)
- ATR stop at 2.2x (slightly wider to avoid premature stops)
- Cleaner code structure with fewer nested conditionals

Why this might beat Sharpe=11.523:
- Z-score filter avoids buying tops/selling bottoms
- Wider ATR stop reduces whipsaw exits
- Cleaner code = fewer bugs in live trading
- 1h timeframe still provides good signal-to-noise ratio
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_zscore_atr_trail_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average - reduces lag vs EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    if half < 1:
        half = 1
    
    def wma(arr, w):
        result = np.zeros(len(arr))
        weights = np.arange(1, w + 1, dtype=np.float64)
        w_sum = np.sum(weights)
        for i in range(w - 1, len(arr)):
            result[i] = np.sum(arr[i - w + 1:i + 1] * weights) / w_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[1:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.zeros(n)
    delta[1:] = np.diff(close)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = (avg_loss > 0) & (~np.isnan(avg_loss))
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = (std > 0) & (~np.isnan(std))
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64) if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    hma_16_1h = calculate_hma(close, period=16)
    hma_48_1h = calculate_hma(close, period=48)
    
    # 4h HMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values.astype(np.float64)
    hma_16_4h = calculate_hma(c_4h, period=16)
    hma_48_4h = calculate_hma(c_4h, period=48)
    
    # 4h trend direction
    trend_4h = np.zeros(len(c_4h))
    for i in range(48, len(c_4h)):
        if hma_16_4h[i] > hma_48_4h[i] and c_4h[i] > hma_16_4h[i]:
            trend_4h[i] = 1
        elif hma_16_4h[i] < hma_48_4h[i] and c_4h[i] < hma_16_4h[i]:
            trend_4h[i] = -1
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.28
    SIZE_HALF = 0.14
    
    # RSI thresholds
    RSI_LONG_ENTRY = 42
    RSI_SHORT_ENTRY = 58
    RSI_EXIT_LONG = 72
    RSI_EXIT_SHORT = 28
    
    # Z-score filter (avoid overextended entries)
    ZSCORE_MAX = 1.5
    
    # ATR stoploss multiplier (slightly wider than before)
    ATR_STOP_MULT = 2.2
    TP_MULT = 2.0
    
    first_valid = max(80, 48, 14, 20)
    
    # Position state tracking
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    trailing_stop = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        z_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Z-score filter - avoid overextended entries
        if abs(z_val) > ZSCORE_MAX:
            if position_side != 0:
                signals[i] = signals[i-1]
            else:
                signals[i] = 0.0
            continue
        
        # Manage existing positions
        if position_side != 0:
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                if lowest_since_entry == 0:
                    lowest_since_entry = price
                else:
                    lowest_since_entry = min(lowest_since_entry, price)
                
                new_trailing = highest_since_entry - ATR_STOP_MULT * atr
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                
                if price < trailing_stop:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    trailing_stop = 0.0
                    continue
                
                tp_price = entry_price + TP_MULT * ATR_STOP_MULT * atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side = 1
                    tp_triggered = True
                    trailing_stop = max(trailing_stop, entry_price)
                    continue
                
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    trailing_stop = 0.0
                    continue
                    
            elif position_side == -1:
                if lowest_since_entry == 0:
                    lowest_since_entry = price
                else:
                    lowest_since_entry = min(lowest_since_entry, price)
                highest_since_entry = max(highest_since_entry, price)
                
                new_trailing = lowest_since_entry + ATR_STOP_MULT * atr
                if trailing_stop == 0 or new_trailing < trailing_stop:
                    trailing_stop = new_trailing
                
                if price > trailing_stop:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    trailing_stop = 0.0
                    continue
                
                tp_price = entry_price - TP_MULT * ATR_STOP_MULT * atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side = -1
                    tp_triggered = True
                    if trailing_stop == 0:
                        trailing_stop = entry_price
                    else:
                        trailing_stop = min(trailing_stop, entry_price)
                    continue
                
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    trailing_stop = 0.0
                    continue
            
            signals[i] = signals[i-1]
            continue
        
        # Entry logic (flat position)
        if trend == 1:
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                signals[i] = SIZE_FULL
                position_side = 1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                trailing_stop = price - ATR_STOP_MULT * atr
            else:
                signals[i] = 0.0
        elif trend == -1:
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                signals[i] = -SIZE_FULL
                position_side = -1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                trailing_stop = price + ATR_STOP_MULT * atr
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals