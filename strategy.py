#!/usr/bin/env python3
"""
EXPERIMENT #003 - MTF Supertrend+MACD+ATR (1h+4h Momentum Trend v1)
==================================================================================================
Hypothesis: Previous strategy timed out due to complex per-bar position state tracking.
This experiment simplifies signal generation while keeping MTF approach:
- Timeframe: 1h entries + 4h trend (proven MTF combination)
- Trend: 4h Supertrend(10,3) for clear trend direction
- Entry: 1h MACD histogram cross + signal line confirmation
- Filter: ATR percentile (avoid extreme volatility regimes)
- Position size: 0.30 discrete levels, stoploss via 2.5*ATR price check
- Simpler logic: vectorized where possible, minimal state tracking

Why this might work better:
- Supertrend is faster to compute than KAMA (no efficiency ratio loops)
- MACD histogram is fully vectorized
- ATR percentile filter replaces BBW+Zscore complexity
- No per-bar position state arrays = faster execution
- Discrete signal levels reduce churn costs
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_atr_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # Fix first element
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend indicator - vectorized where possible"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + mult * atr
    lower_band = hl2 - mult * atr
    
    supertrend[period - 1] = upper_band[period - 1]
    direction[period - 1] = 1
    
    for i in range(period, n):
        if close[i] > supertrend[i - 1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        
        if direction[i] == 1 and close[i] < supertrend[i - 1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        elif direction[i] == -1 and close[i] > supertrend[i - 1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
    
    return supertrend, direction


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (line, signal, histogram) - fully vectorized"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    valid_start = slow + signal - 1
    signal_line[valid_start] = np.mean(macd_line[slow:valid_start + 1])
    
    for i in range(valid_start + 1, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_atr_percentile(atr, period=50):
    """Calculate ATR percentile (rolling) - vectorized"""
    n = len(atr)
    if n < period:
        return np.zeros(n)
    
    percentile = np.zeros(n)
    
    for i in range(period, n):
        window = atr[i - period:i]
        percentile[i] = np.sum(window <= atr[i]) / period
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    if n < 300:
        return signals
    
    # Check if open_time column exists for proper MTF resampling
    if 'open_time' in prices.columns:
        prices_indexed = prices.set_index('open_time')
        
        df_4h = prices_indexed.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(df_4h) < 50:
            return signals
        
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        _, supertrend_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, mult=3.0)
        
        supertrend_dir_4h_series = pd.Series(supertrend_dir_4h, index=df_4h.index)
        supertrend_dir_4h_aligned = supertrend_dir_4h_series.reindex(
            prices_indexed.index, method='ffill'
        ).fillna(0).values
        
    else:
        bars_per_4h = 4
        n_4h = n // bars_per_4h
        
        if n_4h < 50:
            return signals
        
        close_4h = np.array([close[(i + 1) * bars_per_4h - 1] for i in range(n_4h)])
        high_4h = np.array([np.max(high[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        low_4h = np.array([np.min(low[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        
        _, supertrend_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, mult=3.0)
        
        supertrend_dir_4h_aligned = np.zeros(n)
        for i in range(n):
            idx_4h = min(i // bars_per_4h, n_4h - 1)
            supertrend_dir_4h_aligned[i] = supertrend_dir_4h[idx_4h]
    
    # 1h indicators
    atr_1h = calculate_atr(high, low, close, period=14)
    atr_pct_1h = calculate_atr_percentile(atr_1h, period=50)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Thresholds
    ATR_PCT_MIN = 0.20
    ATR_PCT_MAX = 0.85
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 50 * 4, 35 + 9)
    
    # Simple state tracking (minimal for speed)
    in_position = False
    position_side = 0
    entry_price = 0.0
    tp_hit = False
    highest = 0.0
    lowest = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        supertrend_4h = supertrend_dir_4h_aligned[i]
        atr = atr_1h[i]
        atr_pct = atr_pct_1h[i]
        price = close[i]
        hist = macd_hist[i]
        hist_prev = macd_hist[i - 1] if i > 0 else 0
        signal_line = macd_signal[i]
        
        # Stoploss and TP management for existing positions
        if in_position:
            if position_side == 1:
                highest = max(highest, price)
                lowest = min(lowest, price) if lowest > 0 else price
                
                stoploss = entry_price - ATR_STOP_MULT * atr
                if price < stoploss:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_hit = False
                    highest = 0.0
                    lowest = 0.0
                    continue
                
                if not tp_hit:
                    tp_price = entry_price + 2 * ATR_STOP_MULT * atr
                    if price >= tp_price:
                        signals[i] = SIZE_HALF
                        tp_hit = True
                        continue
                
                if tp_hit:
                    trail_stop = highest - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_hit = False
                        highest = 0.0
                        lowest = 0.0
                        continue
            
            elif position_side == -1:
                highest = max(highest, price) if highest > 0 else price
                lowest = min(lowest, price)
                
                stoploss = entry_price + ATR_STOP_MULT * atr
                if price > stoploss:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_hit = False
                    highest = 0.0
                    lowest = 0.0
                    continue
                
                if not tp_hit:
                    tp_price = entry_price - 2 * ATR_STOP_MULT * atr
                    if price <= tp_price:
                        signals[i] = -SIZE_HALF
                        tp_hit = True
                        continue
                
                if tp_hit:
                    trail_stop = lowest + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_hit = False
                        highest = 0.0
                        lowest = 0.0
                        continue
            
            signals[i] = signals[i - 1]
            continue
        
        # ATR volatility filter
        if atr_pct < ATR_PCT_MIN or atr_pct > ATR_PCT_MAX:
            signals[i] = 0.0
            continue
        
        # Long entry: 4h Supertrend bullish + MACD histogram cross up + signal line confirmation
        if supertrend_4h == 1:
            if hist > 0 and hist_prev <= 0 and signal_line > 0:
                signals[i] = SIZE_FULL
                in_position = True
                position_side = 1
                entry_price = price
                tp_hit = False
                highest = price
                lowest = price
                continue
        
        # Short entry: 4h Supertrend bearish + MACD histogram cross down + signal line confirmation
        elif supertrend_4h == -1:
            if hist < 0 and hist_prev >= 0 and signal_line < 0:
                signals[i] = -SIZE_FULL
                in_position = True
                position_side = -1
                entry_price = price
                tp_hit = False
                highest = price
                lowest = price
                continue
        
        signals[i] = 0.0
    
    return signals