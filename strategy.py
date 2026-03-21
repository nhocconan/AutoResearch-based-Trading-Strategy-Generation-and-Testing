#!/usr/bin/env python3
"""
EXPERIMENT #007 - HMA Trend + MACD Histogram Entry + ADX Strength Filter
===============================================================================
Hypothesis: Using HMA (smoother than EMA/KAMA) for 4H trend direction, combined with
MACD histogram momentum for 1H entry timing, and ADX filter to avoid weak/choppy markets.
This differs from current best (KAMA+MACD+ADX) by using HMA for less lag and proper
multi-timeframe structure with discrete position sizing.

Key innovations:
- HMA(21/42) for 4H trend - smoother than KAMA with less lag
- MACD histogram slope for entry timing (not just crossover)
- ADX(14) > 25 filter to avoid trading in choppy markets
- Discrete position sizing (0.0, ±0.25, ±0.35) to reduce churn costs
- ATR trailing stop at 2.5*ATR with proper entry price tracking
- Z-score filter to avoid entering at extreme deviations (>2 std)

Why this might beat Sharpe=2.139:
- HMA reduces whipsaw better than KAMA in trending markets
- MACD histogram slope catches momentum earlier than crossover
- ADX filter prevents trading during low-volatility chop
- Multi-timeframe logic proven in mtf_hma_rsi_zscore_v1 (Sharpe=5.4)
"""

import numpy as np
import pandas as pd

name = "mtf_hma_macd_adx_zscore_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculations
    def wma(data, w):
        result = np.zeros(len(data))
        for i in range(w - 1, len(data)):
            weights = np.arange(1, w + 1)
            result[i] = np.sum(data[i - w + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.zeros(n)
    raw_hma = 2 * wma_half - wma_full
    for i in range(sqrt_period - 1, n):
        hma[i] = wma(raw_hma, sqrt_period)[i]
    
    return hma


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    # EMA calculations
    def ema(data, period):
        result = np.zeros(n)
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, n):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = np.zeros(n)
    multiplier = 2 / (signal + 1)
    first_valid = fast + signal - 1
    if first_valid < n:
        signal_line[first_valid] = np.mean(macd_line[fast:first_valid + 1])
        for i in range(first_valid + 1, n):
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    atr = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_di[period - 1] = 100 * np.mean(plus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    minus_di[period - 1] = 100 * np.mean(minus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_di[i - 1] * (period - 1) / 100 * atr[i - 1] + plus_dm[i]) / atr[i]) if atr[i] > 0 else 0
        minus_di[i] = 100 * ((minus_di[i - 1] * (period - 1) / 100 * atr[i - 1] + minus_dm[i]) / atr[i]) if atr[i] > 0 else 0
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


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
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_1h, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # 4h HMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # Calculate 4h HMA for trend
    hma_4h_fast = calculate_hma(c_4h, period=21)
    hma_4h_slow = calculate_hma(c_4h, period=42)
    
    # 4h trend direction based on HMA crossover
    trend_4h = np.zeros(len(c_4h))
    for i in range(42, len(c_4h)):
        if hma_4h_fast[i] > hma_4h_slow[i]:
            trend_4h[i] = 1  # Bullish
        elif hma_4h_fast[i] < hma_4h_slow[i]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.25   # Reduced position in marginal conditions
    
    # ADX threshold for trend strength
    ADX_MIN = 25       # Only trade when ADX > 25 (strong trend)
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 2.0   # Don't enter if price > 2 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 42, 26, 28)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_price = np.zeros(n)  # For trailing stop on longs
    lowest_price = np.zeros(n)   # For trailing stop on shorts
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(macd_hist[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        adx_val = adx_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        macd_histogram = macd_hist[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ADX filter - only trade in strong trends
        if adx_val < ADX_MIN:
            # If we have a position, hold it; otherwise stay flat
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                highest_price[i] = max(highest_price[i - 1], price)
                lowest_price[i] = min(lowest_price[i - 1], price)
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_highest = highest_price[i - 1] if highest_price[i - 1] > 0 else price
            prev_lowest = lowest_price[i - 1] if lowest_price[i - 1] > 0 else price
            
            if prev_side == 1:  # Long position
                # Update highest price for trailing stop
                current_highest = max(prev_highest, price)
                highest_price[i] = current_highest
                stoploss_price = current_highest - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    lowest_price[i] = 0
                    continue
            elif prev_side == -1:  # Short position
                # Update lowest price for trailing stop
                current_lowest = min(prev_lowest, price)
                lowest_price[i] = current_lowest
                stoploss_price = current_lowest + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    lowest_price[i] = 0
                    continue
        
        # Z-score filter - avoid entering at extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            # If we have a position, hold it; otherwise stay flat
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                if position_side[i] == 1:
                    highest_price[i] = max(highest_price[i - 1], price)
                    lowest_price[i] = lowest_price[i - 1]
                else:
                    highest_price[i] = highest_price[i - 1]
                    lowest_price[i] = min(lowest_price[i - 1], price)
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # MACD histogram slope for entry timing
        macd_slope = 0
        if i >= 2 and not np.isnan(macd_hist[i - 1]):
            macd_slope = macd_histogram - macd_hist[i - 1]
        
        if trend == 1:  # 4h uptrend
            # MACD histogram positive and rising for long entry
            if macd_histogram > 0 and macd_slope > 0:
                # Strong momentum - full position
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                highest_price[i] = price
                lowest_price[i] = price
            elif macd_histogram > 0:
                # Positive but not rising - half position
                signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
                highest_price[i] = price
                lowest_price[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    highest_price[i] = max(highest_price[i - 1], price)
                    lowest_price[i] = lowest_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    lowest_price[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # MACD histogram negative and falling for short entry
            if macd_histogram < 0 and macd_slope < 0:
                # Strong momentum - full short
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                highest_price[i] = price
                lowest_price[i] = price
            elif macd_histogram < 0:
                # Negative but not falling - half short
                signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
                highest_price[i] = price
                lowest_price[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    highest_price[i] = highest_price[i - 1]
                    lowest_price[i] = min(lowest_price[i - 1], price)
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    lowest_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_price[i] = 0
            lowest_price[i] = 0
    
    return signals