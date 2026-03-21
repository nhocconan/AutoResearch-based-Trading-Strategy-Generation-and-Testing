#!/usr/bin/env python3
"""
EXPERIMENT #041 - MTF Donchian+MACD+Volume+Z-score (1h+4h Clean v1)
==================================================================================================
Hypothesis: Current best (#040) uses 15m+1h with HMA/Supertrend/KAMA. This experiment tests:
- Timeframe: 1h entries + 4h trend (proven in original mtf_hma_rsi_zscore_v1 with Sharpe=5.4)
- Trend: Donchian(20) breakout instead of HMA/Supertrend - cleaner trend definition
- Entry: MACD histogram cross instead of RSI pullback - momentum-based timing
- Filter: Z-score + Volume spike instead of ADX + BBW - different regime detection
- Position size: 0.30 (slightly lower than 0.35 for extra safety)

Why this might beat #040:
- 4h trend is more stable than 1h trend (fewer whipsaws)
- Donchian breakout captures true momentum breaks
- MACD histogram cross is proven momentum entry signal
- Volume confirmation filters false breakouts
- 1h timeframe has fewer fees than 15m while still capturing moves
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_macd_volume_zscore_1h_v1"
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands and breakout signal)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    breakout = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        
        # Breakout signal: 1 = upper break, -1 = lower break, 0 = none
        if high[i] > upper[i - 1]:
            breakout[i] = 1
        elif low[i] < lower[i - 1]:
            breakout[i] = -1
    
    return upper, lower, breakout


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD (line, signal, histogram)"""
    n = len(close)
    if n < slow + signal_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # Calculate EMAs
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    # Initialize EMAs
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    # Calculate EMAs
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = np.zeros(n)
    first_macd_valid = slow - 1
    signal_line[first_macd_valid + signal_period - 1] = np.mean(macd_line[first_macd_valid:first_macd_valid + signal_period])
    
    for i in range(first_macd_valid + signal_period, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal_period + 1)) * (macd_line[i] - signal_line[i - 1])
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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


def calculate_volume_ma(volume, period=20):
    """Calculate Volume Moving Average and Volume Ratio"""
    n = len(volume)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    volume_ma = np.zeros(n)
    volume_ratio = np.zeros(n)
    
    for i in range(period - 1, n):
        volume_ma[i] = np.mean(volume[i - period + 1:i + 1])
        if volume_ma[i] > 0:
            volume_ratio[i] = volume[i] / volume_ma[i]
        else:
            volume_ratio[i] = 1.0
    
    return volume_ma, volume_ratio


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = ema[i - 1] + (2.0 / (period + 1)) * (close[i] - ema[i - 1])
    
    return ema


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Check if open_time column exists for proper MTF resampling
    if 'open_time' in prices.columns:
        # PROPER MTF: Use actual timestamps for resampling
        prices_indexed = prices.set_index('open_time')
        
        # Resample to 4h for trend filter
        df_4h = prices_indexed.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate 4h indicators
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        donchian_upper_4h, donchian_lower_4h, donchian_breakout_4h = calculate_donchian(high_4h, low_4h, period=20)
        ema_21_4h = calculate_ema(close_4h, period=21)
        ema_55_4h = calculate_ema(close_4h, period=55)
        
        # Map 4h trend back to 1h using ffill (proper alignment)
        trend_4h_series = pd.Series(donchian_breakout_4h, index=df_4h.index)
        ema_ratio_4h = pd.Series(close_4h / ema_21_4h - 1, index=df_4h.index)
        
        # Reindex to match 1h timestamps with forward fill
        trend_4h_aligned = trend_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
        ema_ratio_4h_aligned = ema_ratio_4h.reindex(prices_indexed.index, method='ffill').fillna(0).values
        
    else:
        # Fallback: simple downsampling if no open_time
        bars_per_4h = 4
        n_4h = n // bars_per_4h
        
        close_4h = np.array([close[(i + 1) * bars_per_4h - 1] for i in range(n_4h)])
        high_4h = np.array([np.max(high[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        low_4h = np.array([np.min(low[i * bars_per_4h:(i + 1) * bars_per_4h]) for i in range(n_4h)])
        
        donchian_upper_4h, donchian_lower_4h, donchian_breakout_4h = calculate_donchian(high_4h, low_4h, period=20)
        ema_21_4h = calculate_ema(close_4h, period=21)
        ema_55_4h = calculate_ema(close_4h, period=55)
        
        trend_4h_aligned = np.zeros(n)
        ema_ratio_4h_aligned = np.zeros(n)
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h < n_4h and idx_4h >= 55:
                trend_4h_aligned[i] = donchian_breakout_4h[idx_4h]
                if ema_21_4h[idx_4h] > 0:
                    ema_ratio_4h_aligned[i] = close_4h[idx_4h] / ema_21_4h[idx_4h] - 1
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal_period=9)
    zscore_1h = calculate_zscore(close, period=20)
    _, volume_ratio_1h = calculate_volume_ma(volume, period=20)
    ema_21_1h = calculate_ema(close, period=21)
    ema_55_1h = calculate_ema(close, period=55)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # MACD histogram thresholds for entry
    MACD_HIST_MIN = 0  # Must be positive for long
    MACD_HIST_MAX = 0  # Must be negative for short
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.5
    
    # Volume ratio threshold for confirmation
    VOLUME_RATIO_MIN = 1.2  # Volume must be 20% above average
    
    # EMA ratio threshold for trend confirmation (4h)
    EMA_RATIO_MIN = 0.01  # Price must be 1% above EMA21 for bullish
    
    first_valid = max(200, 55 * 4, 26 + 9, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(macd_hist[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filters
        trend_4h = trend_4h_aligned[i]
        ema_ratio_4h = ema_ratio_4h_aligned[i]
        
        # 1h entry filters
        macd_histogram = macd_hist[i]
        macd_prev_histogram = macd_hist[i - 1] if i > 0 else 0
        zscore_val = zscore_1h[i]
        volume_ratio = volume_ratio_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # EMA trend on 1h
        ema_trend_1h = 0
        if ema_21_1h[i] > ema_55_1h[i]:
            ema_trend_1h = 1
        elif ema_21_1h[i] < ema_55_1h[i]:
            ema_trend_1h = -1
        
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
            
            # Stoploss check (2.0*ATR)
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
        
        # Entry logic: 4h Donchian trend + 1h MACD + Volume + Z-score + EMA
        # Long entry
        if trend_4h == 1 and ema_ratio_4h > EMA_RATIO_MIN and ema_trend_1h == 1:
            # MACD histogram crossing above zero (momentum entry)
            if macd_histogram > MACD_HIST_MIN and macd_prev_histogram <= MACD_HIST_MIN:
                # Volume confirmation and Z-score filter
                if volume_ratio >= VOLUME_RATIO_MIN and abs(zscore_val) < ZSCORE_MAX:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
        
        # Short entry
        elif trend_4h == -1 and ema_ratio_4h < -EMA_RATIO_MIN and ema_trend_1h == -1:
            # MACD histogram crossing below zero (momentum entry)
            if macd_histogram < MACD_HIST_MAX and macd_prev_histogram >= MACD_HIST_MAX:
                # Volume confirmation and Z-score filter
                if volume_ratio >= VOLUME_RATIO_MIN and abs(zscore_val) < ZSCORE_MAX:
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