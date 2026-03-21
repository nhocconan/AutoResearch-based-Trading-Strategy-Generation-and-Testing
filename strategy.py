#!/usr/bin/env python3
"""
EXPERIMENT #116 - MTF HMA+RSI+Chandelier+VolRegime+Zscore (15m+4h Optimized v1)
==================================================================================================
Hypothesis: Building on #112 (Sharpe=6.193) and #108 (Sharpe=7.706), combine:
- 15m entries with 4h trend filter (proven MTF from winning strategies)
- HMA(21/63) for smooth trend detection
- RSI(14) pullback entries in trending markets
- Chandelier Exit (ATR(22)*3) for trailing stoploss
- Volatility regime adjustment: reduce position size when ATR percentile > 70th
- Z-score(20) filter to avoid extreme entries
- Discrete position sizing: 0.0, ±0.20, ±0.35 (CRITICAL for drawdown control)

Why this should beat current best (Sharpe=16.016):
- Better Chandelier implementation (22-period ATR vs 14)
- Volatility-adjusted sizing reduces risk in high vol regimes
- Cleaner MTF resampling with proper 4h alignment (16 x 15m = 4h)
- Z-score filter prevents chasing extremes
- Based on proven winning combinations from #105, #108, #112
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_chandelier_volregime_zscore_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def calculate_chandelier_exit(high, low, close, atr_period=22, multiplier=3.0):
    """
    Calculate Chandelier Exit (ATR trailing stop)
    Long exit: highest_high - multiplier * ATR
    Short exit: lowest_low + multiplier * ATR
    """
    n = len(close)
    if n < atr_period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, atr_period)
    
    chandelier_long = np.zeros(n)  # Stop level for long positions
    chandelier_short = np.zeros(n)  # Stop level for short positions
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    
    for i in range(atr_period - 1, n):
        # Track highest high and lowest low over lookback
        if i == atr_period - 1:
            highest_high[i] = np.max(high[:i + 1])
            lowest_low[i] = np.min(low[:i + 1])
        else:
            highest_high[i] = max(highest_high[i - 1], high[i])
            lowest_low[i] = min(lowest_low[i - 1], low[i])
        
        chandelier_long[i] = highest_high[i] - multiplier * atr[i]
        chandelier_short[i] = lowest_low[i] + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile for volatility regime detection"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        sorted_window = np.sort(window)
        rank = np.searchsorted(sorted_window, atr[i])
        percentile[i] = rank / lookback
    
    return percentile


def resample_to_4h(close, high, low, bars_per_4h=16):
    """Resample 15m data to 4h (16 x 15m = 4h)"""
    n = len(close)
    n_4h = n // bars_per_4h
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    return c_4h, h_4h, l_4h, n_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m_fast = calculate_hma(close, period=21)
    hma_15m_slow = calculate_hma(close, period=63)
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(
        high, low, close, atr_period=22, multiplier=3.0
    )
    atr_pct_15m = calculate_atr_percentile(atr_15m, lookback=100)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h, n_4h = resample_to_4h(close, high, low, bars_per_4h)
    
    # 4h indicators for trend
    hma_4h_fast = calculate_hma(c_4h, period=21)
    hma_4h_slow = calculate_hma(c_4h, period=63)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    chandelier_long_4h, chandelier_short_4h = calculate_chandelier_exit(
        h_4h, l_4h, c_4h, atr_period=22, multiplier=3.0
    )
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    chandelier_long_4h_mapped = np.zeros(n)
    chandelier_short_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 63:
            # 4h trend: fast HMA > slow HMA = bullish
            if c_4h[idx_4h] > hma_4h_fast[idx_4h] and hma_4h_fast[idx_4h] > hma_4h_slow[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h_fast[idx_4h] and hma_4h_fast[idx_4h] < hma_4h_slow[idx_4h]:
                trend_4h[i] = -1
            
            chandelier_long_4h_mapped[i] = chandelier_long_4h[idx_4h]
            chandelier_short_4h_mapped[i] = chandelier_short_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.0875
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ATR stoploss multiplier (Chandelier uses 3.0)
    CHANDELIER_MULT = 3.0
    
    # Volatility regime thresholds
    VOL_HIGH_THRESHOLD = 0.70  # Reduce size when ATR percentile > 70%
    VOL_LOW_THRESHOLD = 0.30   # Full size when ATR percentile < 30%
    
    first_valid = max(200, 63 * bars_per_4h, 100, 22)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    chandelier_stop = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        atr_pct = atr_pct_15m[i]
        ch_long_4h = chandelier_long_4h_mapped[i]
        ch_short_4h = chandelier_short_4h_mapped[i]
        
        # Volatility regime adjustment for position sizing
        if atr_pct > VOL_HIGH_THRESHOLD:
            current_size = SIZE_QUARTER  # High vol: reduce to 25%
        elif atr_pct < VOL_LOW_THRESHOLD:
            current_size = SIZE_FULL  # Low vol: full size
        else:
            current_size = SIZE_HALF  # Medium vol: half size
        
        # Check Chandelier exit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            prev_chandelier = chandelier_stop[i - 1]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
                # Update Chandelier stop (trailing)
                new_chandelier = current_high - CHANDELIER_MULT * atr
                chandelier_stop[i] = max(prev_chandelier, new_chandelier) if prev_chandelier > 0 else new_chandelier
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
                # Update Chandelier stop (trailing)
                new_chandelier = current_low + CHANDELIER_MULT * atr
                chandelier_stop[i] = min(prev_chandelier, new_chandelier) if prev_chandelier > 0 else new_chandelier
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Chandelier stoploss check
            if prev_side == 1:
                if price < chandelier_stop[i]:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    continue
                
                # Take profit check (2R based on initial ATR)
                initial_r = CHANDELIER_MULT * atr
                tp_price = prev_entry + 2 * initial_r
                if not prev_tp and price >= tp_price:
                    signals[i] = current_size / 2
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_high - CHANDELIER_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        chandelier_stop[i] = 0
                        continue
                
            elif prev_side == -1:
                if price > chandelier_stop[i]:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    continue
                
                # Take profit check (2R based on initial ATR)
                initial_r = CHANDELIER_MULT * atr
                tp_price = prev_entry - 2 * initial_r
                if not prev_tp and price <= tp_price:
                    signals[i] = -current_size / 2
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_low + CHANDELIER_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        chandelier_stop[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            chandelier_stop[i] = chandelier_stop[i - 1]
            continue
        
        # Entry logic: 4h HMA trend + 15m RSI pullback + Z-score filter
        if trend == 1:  # Bullish trend on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX and
                price > ch_long_4h):  # Above 4h Chandelier support
                signals[i] = current_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                chandelier_stop[i] = price - CHANDELIER_MULT * atr
                
        elif trend == -1:  # Bearish trend on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX and
                price < ch_short_4h):  # Below 4h Chandelier resistance
                signals[i] = -current_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                chandelier_stop[i] = price + CHANDELIER_MULT * atr
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals