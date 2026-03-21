#!/usr/bin/env python3
"""
EXPERIMENT #011 - MTF Donchian+DEMA+MACD+Volume+Z-score (4h+1h v1)
==================================================================================================
Hypothesis: Combine proven elements from #005 (DEMA+Supertrend+MACD, Sharpe=0.213) and 
#006 (KAMA+Donchian+RSI, Sharpe=0.160) with volume confirmation (new element).

Key changes:
- Timeframe: 1h (stable, proven in winning strategies)
- MTF: 4h Donchian trend + 1h DEMA entry (cleaner than 15m+1h complexity)
- Trend: Donchian Channel 20-period breakout (strong trend signal)
- Entry: DEMA(8/21) crossover with MACD histogram confirmation
- Filters: Z-score < 2.0, Volume > 20-period SMA (new confirmation)
- Position size: 0.30 (conservative, proven safe range)
- Stoploss: 2.5*ATR (slightly wider for 1h timeframe)

Why this should beat #005 (Sharpe=0.213):
- Donchian trend is cleaner than HMA for trend direction
- Volume filter adds confirmation (reduces false signals)
- DEMA is faster than HMA for entries
- Simpler MTF logic (4h+1h vs 15m+1h) reduces bugs
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_dema_macd_volume_zscore_4h_1h_v1"
timeframe = "1h"
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


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    dema = np.zeros(n)
    
    multiplier = 2.0 / (period + 1)
    
    ema1[0] = close[0]
    for i in range(1, n):
        ema1[i] = multiplier * close[i] + (1 - multiplier) * ema1[i - 1]
    
    ema2[0] = ema1[0]
    for i in range(1, n):
        ema2[i] = multiplier * ema1[i] + (1 - multiplier) * ema2[i - 1]
    
    for i in range(n):
        dema[i] = 2 * ema1[i] - ema2[i]
    
    return dema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    n = len(close)
    if n < slow:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    fast_mult = 2.0 / (fast + 1)
    slow_mult = 2.0 / (slow + 1)
    signal_mult = 2.0 / (signal + 1)
    
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    for i in range(1, n):
        ema_fast[i] = fast_mult * close[i] + (1 - fast_mult) * ema_fast[i - 1]
        ema_slow[i] = slow_mult * close[i] + (1 - slow_mult) * ema_slow[i - 1]
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    signal_line[slow - 1] = macd_line[slow - 1]
    for i in range(slow, n):
        signal_line[i] = signal_mult * macd_line[i] + (1 - signal_mult) * signal_line[i - 1]
    
    for i in range(n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)"""
    n = len(close := high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower


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


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    volume_sma = np.zeros(n)
    for i in range(period - 1, n):
        volume_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return volume_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    dema_fast_1h = calculate_dema(close, period=8)
    dema_slow_1h = calculate_dema(close, period=21)
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Resample to 4h for trend filter using proper method
    prices_indexed = prices.set_index('open_time')
    
    # Resample to 4h
    df_4h = prices_indexed.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian trend
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    donchian_upper_4h, donchian_middle_4h, donchian_lower_4h = calculate_donchian(high_4h, low_4h, period=20)
    
    # Determine 4h trend direction
    trend_4h = np.zeros(len(close_4h))
    for i in range(20, len(close_4h)):
        if close_4h[i] > donchian_middle_4h[i]:
            trend_4h[i] = 1
        elif close_4h[i] < donchian_middle_4h[i]:
            trend_4h[i] = -1
    
    # Map 4h trend back to 1h using reindex with ffill
    trend_4h_series = pd.Series(trend_4h, index=df_4h.index)
    trend_4h_aligned = trend_4h_series.reindex(prices_indexed.index, method='ffill').values
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Volume confirmation threshold
    VOLUME_MULT = 1.0  # Volume must be >= 1.0x SMA
    
    first_valid = max(200, 20 * 4, 26, 20)  # Need enough data for all indicators
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h_aligned[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol = volume[i]
        vol_sma = volume_sma_1h[i]
        
        # Volume filter - must have sufficient volume
        if vol_sma > 0 and vol < VOLUME_MULT * vol_sma:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Z-score filter - avoid extreme moves
        if abs(zscore_val) >= ZSCORE_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h Donchian trend + 1h DEMA crossover + MACD confirmation
        dema_cross = dema_fast_1h[i] - dema_slow_1h[i]
        dema_cross_prev = dema_fast_1h[i - 1] - dema_slow_1h[i - 1] if i > 0 else 0
        macd_hist = macd_hist_1h[i]
        macd_hist_prev = macd_hist_1h[i - 1] if i > 0 else 0
        
        # Bullish entry: 4h uptrend + DEMA golden cross + MACD histogram positive/increasing
        if trend == 1 and dema_cross > 0 and dema_cross_prev <= 0 and macd_hist > 0:
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Bearish entry: 4h downtrend + DEMA death cross + MACD histogram negative/decreasing
        elif trend == -1 and dema_cross < 0 and dema_cross_prev >= 0 and macd_hist < 0:
            signals[i] = -SIZE_FULL
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals