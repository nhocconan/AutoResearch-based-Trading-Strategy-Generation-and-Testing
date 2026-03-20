#!/usr/bin/env python3
"""
EXPERIMENT #005 - Multi-Timeframe DEMA Trend + MACD Entry + BB Regime Filter
=============================================================================
Hypothesis: 4h DEMA(21/55) crossover provides faster trend detection than HMA,
combined with 1h MACD histogram for precise entry timing. Bollinger Band width
percentile filters out extreme volatility regimes (squeeze/expansion).

Key improvements over mtf_hma_rsi_zscore_v1:
- DEMA double-smooths EMA for less lag than HMA
- MACD histogram cross gives clearer entry signals than RSI levels
- BB width percentile adapts to volatility regime dynamically
- ATR trailing stop: signal→0 when price moves 2*ATR against position

Why this might beat Sharpe=1.768:
- DEMA responds faster to trend changes than HMA/Supertrend
- MACD histogram momentum confirms entries (not just pullback levels)
- BB regime filter avoids trading during abnormal volatility
- Proper stoploss implementation reduces drawdown
"""

import numpy as np
import pandas as pd

name = "mtf_dema_macd_bbregime_v1"
timeframe = "1h"
leverage = 1.0


def calculate_dema(close, period=21):
    """
    Calculate Double Exponential Moving Average
    DEMA = 2*EMA(n) - EMA(EMA(n))
    Reduces lag significantly vs standard EMA
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = np.nan
    
    return dema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    close_series = pd.Series(close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    
    macd_series = pd.Series(macd_line)
    signal_line = macd_series.ewm(span=signal, adjust=False, min_periods=signal).mean().values
    
    histogram = macd_line - signal_line
    
    # Set initial values to nan
    macd_line[:slow] = np.nan
    signal_line[:slow+signal] = np.nan
    histogram[:slow+signal] = np.nan
    
    return macd_line, signal_line, histogram


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    return atr


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth"""
    n = len(close)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    # Bandwidth = (Upper - Lower) / Middle
    bandwidth = np.zeros(n)
    mask = middle > 0
    bandwidth[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, lower, middle, bandwidth


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank of a series"""
    n = len(series)
    result = np.zeros(n)
    
    for i in range(window-1, n):
        if not np.isnan(series[i]):
            window_data = series[i-window+1:i+1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                result[i] = np.sum(window_data < series[i]) / len(window_data)
    
    return result


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h Indicators for Entry Timing ==========
    dema_21_1h = calculate_dema(close, period=21)
    dema_55_1h = calculate_dema(close, period=55)
    
    macd_line_1h, signal_line_1h, histogram_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    
    atr_1h = calculate_atr(high, low, close, period=14)
    
    _, _, bb_middle_1h, bb_bandwidth_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_percentile_1h = calculate_percentile_rank(bb_bandwidth_1h, window=100)
    
    # ========== 4h Trend Filter (resample 1h → 4h) ==========
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
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    n_4h = len(close_4h)
    
    # 4h DEMA crossover for trend
    dema_21_4h = calculate_dema(close_4h, period=21)
    dema_55_4h = calculate_dema(close_4h, period=55)
    
    # 4h ATR for stoploss
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
    
    # Determine 4h trend direction
    trend_4h = np.zeros(n_4h)
    for i in range(n_4h):
        if not np.isnan(dema_21_4h[i]) and not np.isnan(dema_55_4h[i]):
            if dema_21_4h[i] > dema_55_4h[i]:
                trend_4h[i] = 1  # Bullish
            elif dema_21_4h[i] < dema_55_4h[i]:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Map 4h ATR to 1h (scale by 2 since 4h = 4x 1h bars, sqrt(4)=2)
    atr_1h_mapped = np.zeros(n)
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(atr_4h):
            atr_1h_mapped[i] = atr_4h[idx_4h] / 2.0
    
    # ========== Generate Signals ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.20
    
    # MACD histogram thresholds
    MACD_LONG_THRESHOLD = 0.0    # Histogram crosses above zero
    MACD_SHORT_THRESHOLD = 0.0   # Histogram crosses below zero
    
    # BB regime filter (percentile)
    BB_REGIME_MIN = 0.20  # Don't trade in extreme squeeze (<20th percentile)
    BB_REGIME_MAX = 0.85  # Don't trade in extreme expansion (>85th percentile)
    
    # Wait for all indicators to be valid
    first_valid = max(55, 26+9, 14, 20, 100)
    
    # Track entry prices for stoploss
    entry_price = np.zeros(n)
    position_direction = np.zeros(n)  # 1=long, -1=short, 0=none
    
    for i in range(first_valid, n):
        # Check for valid data
        if (np.isnan(dema_21_1h[i]) or np.isnan(dema_55_1h[i]) or
            np.isnan(histogram_1h[i]) or np.isnan(atr_1h[i]) or
            np.isnan(bb_percentile_1h[i]) or np.isnan(trend_1h[i]) or
            np.isnan(atr_1h_mapped[i])):
            signals[i] = 0.0
            position_direction[i] = 0
            continue
        
        trend = trend_1h[i]
        macd_hist = histogram_1h[i]
        bb_regime = bb_percentile_1h[i]
        atr_val = atr_1h_mapped[i]
        
        # BB regime filter - avoid extreme volatility
        if bb_regime < BB_REGIME_MIN or bb_regime > BB_REGIME_MAX:
            signals[i] = 0.0
            position_direction[i] = 0
            continue
        
        # Check ATR trailing stoploss (2*ATR against position)
        if position_direction[i-1] != 0 and i > 0:
            prev_direction = position_direction[i-1]
            prev_entry = entry_price[i-1] if entry_price[i-1] > 0 else close[i-1]
            
            if prev_direction == 1:  # Long position
                stop_loss = prev_entry - 2 * atr_val
                if close[i] < stop_loss:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    entry_price[i] = 0
                    continue
            elif prev_direction == -1:  # Short position
                stop_loss = prev_entry + 2 * atr_val
                if close[i] > stop_loss:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    entry_price[i] = 0
                    continue
        
        # Generate new signals based on trend + MACD
        if trend == 1:  # 4h uptrend - look for long entries
            if macd_hist > MACD_LONG_THRESHOLD:
                # MACD bullish - enter long
                if position_direction[i-1] <= 0:
                    signals[i] = SIZE_FULL
                    position_direction[i] = 1
                    entry_price[i] = close[i]
                else:
                    signals[i] = SIZE_FULL
                    position_direction[i] = 1
            elif macd_hist < 0 and position_direction[i-1] == 1:
                # MACD turning bearish - reduce position
                signals[i] = SIZE_HALF
                position_direction[i] = 1
            else:
                signals[i] = 0.0
                position_direction[i] = 0
                entry_price[i] = 0
                
        elif trend == -1:  # 4h downtrend - look for short entries
            if macd_hist < MACD_SHORT_THRESHOLD:
                # MACD bearish - enter short
                if position_direction[i-1] >= 0:
                    signals[i] = -SIZE_FULL
                    position_direction[i] = -1
                    entry_price[i] = close[i]
                else:
                    signals[i] = -SIZE_FULL
                    position_direction[i] = -1
            elif macd_hist > 0 and position_direction[i-1] == -1:
                # MACD turning bullish - reduce position
                signals[i] = -SIZE_HALF
                position_direction[i] = -1
            else:
                signals[i] = 0.0
                position_direction[i] = 0
                entry_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_direction[i] = 0
            entry_price[i] = 0
    
    # Smooth signal transitions to reduce churn
    for i in range(1, n):
        if signals[i] != signals[i-1] and signals[i] != 0 and signals[i-1] != 0:
            # Only allow transitions through zero or between discrete levels
            if abs(signals[i]) != abs(signals[i-1]):
                # Keep previous signal if change is too small
                if abs(signals[i] - signals[i-1]) < 0.10:
                    signals[i] = signals[i-1]
    
    return signals