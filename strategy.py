#!/usr/bin/env python3
"""
EXPERIMENT #007 - Multi-Timeframe Supertrend + MACD + ADX + BB Regime
======================================================================
Hypothesis: Combining 4h Supertrend trend filter with 1h MACD histogram 
entry signals, ADX strength filter, and Bollinger Band regime detection 
will improve risk-adjusted returns over HMA+RSI+Z-score.

Key differences from mtf_hma_rsi_zscore_v1:
- Supertrend(10,3) instead of HMA for clearer trend signals
- MACD histogram cross for momentum entry timing
- ADX(14) > 25 filter to avoid weak/choppy trends
- Bollinger Band Width percentile for volatility regime
- ATR(14) trailing stop: signal→0 when price moves 2*ATR against position

Why this might beat Sharpe=5.4:
- Supertrend provides clearer trend direction with built-in stop
- MACD histogram captures momentum shifts better than RSI
- ADX filter avoids trading in choppy markets (major drawdown source)
- BB regime helps avoid entries during extreme volatility
- ATR stoploss protects against adverse moves

Position sizing: 0.20-0.35 discrete levels
Stoploss: signal→0 when price moves 2*ATR against position
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_bbadx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
            
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1 if close[i] < upper_band[i] else 1
        else:
            # Update upper/lower bands based on previous trend
            if trend[i-1] == 1:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            else:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
            
            # Determine trend
            if close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            elif close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                trend[i] = trend[i-1]
    
    return supertrend, trend


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    # Band width as percentage of middle band
    band_width = np.zeros(n)
    mask = middle > 0
    band_width[mask] = (upper[mask] - lower[mask]) / middle[mask] * 100
    
    return upper, middle, lower, band_width


def calculate_atr_stoploss(close, high, low, atr, position, entry_price, multiplier=2):
    """
    Calculate ATR-based stoploss level
    Returns True if stoploss triggered (signal should go to 0)
    """
    n = len(close)
    stop_triggered = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if position[i-1] == 0:
            continue
            
        if np.isnan(atr[i]):
            continue
        
        if position[i-1] > 0:  # Long position
            stop_level = entry_price - multiplier * atr[i]
            if close[i] < stop_level:
                stop_triggered[i] = True
        elif position[i-1] < 0:  # Short position
            stop_level = entry_price + multiplier * atr[i]
            if close[i] > stop_level:
                stop_triggered[i] = True
    
    return stop_triggered


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, period=20)
    
    # Calculate BB width percentile for regime detection
    bb_width_percentile = np.zeros(n)
    for i in range(20, n):
        if np.isnan(bb_width[i]):
            continue
        window = bb_width[max(0, i-50):i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            bb_width_percentile[i] = np.sum(valid_window <= bb_width[i]) / len(valid_window)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
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
    
    # Calculate 4h Supertrend
    supertrend_4h, trend_4h = calculate_supertrend(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=10,
        multiplier=3
    )
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    positions = np.zeros(n)  # Track position for stoploss
    entry_prices = np.zeros(n)  # Track entry price for stoploss
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # MACD thresholds for momentum entry
    MACD_LONG_THRESHOLD = 0.0    # Histogram crossing above 0
    MACD_SHORT_THRESHOLD = 0.0   # Histogram crossing below 0
    
    # ADX threshold for trend strength
    ADX_MIN = 25  # Only trade when ADX > 25 (strong trend)
    
    # BB width percentile thresholds
    BB_WIDTH_LOW = 0.3   # Don't trade during extreme squeeze
    BB_WIDTH_HIGH = 0.9  # Don't trade during extreme expansion
    
    first_valid = max(48, 26, 20, 14)  # Wait for all indicators
    
    for i in range(first_valid, n):
        # Check for NaN values
        if (np.isnan(trend_1h[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(adx_1h[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(atr_1h[i])):
            signals[i] = 0.0
            positions[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_histogram = macd_hist[i]
        adx_val = adx_1h[i]
        bb_percentile = bb_width_percentile[i]
        
        # Check ATR stoploss first (overrides all other signals)
        if positions[i-1] != 0:
            stop_triggered = calculate_atr_stoploss(
                close[:i+1], high[:i+1], low[:i+1], 
                atr_1h[:i+1], positions[:i], entry_prices[i-1], 
                multiplier=2
            )
            if stop_triggered[-1]:
                signals[i] = 0.0
                positions[i] = 0.0
                entry_prices[i] = 0.0
                continue
        
        # ADX filter - only trade strong trends
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            if positions[i-1] != 0:
                positions[i] = 0.0
                entry_prices[i] = 0.0
            else:
                positions[i] = 0.0
            continue
        
        # BB regime filter - avoid extreme volatility regimes
        if bb_percentile < BB_WIDTH_LOW or bb_percentile > BB_WIDTH_HIGH:
            signals[i] = 0.0
            if positions[i-1] != 0:
                positions[i] = 0.0
                entry_prices[i] = 0.0
            else:
                positions[i] = 0.0
            continue
        
        # MACD histogram momentum filter
        macd_prev = macd_hist[i-1] if i > 0 else 0
        
        if trend == 1:  # 4h uptrend
            # Look for MACD histogram crossing above 0 or positive and rising
            if macd_histogram > MACD_LONG_THRESHOLD:
                if macd_histogram > macd_prev:  # Momentum increasing
                    signals[i] = SIZE_FULL
                    if positions[i-1] == 0:
                        positions[i] = SIZE_FULL
                        entry_prices[i] = close[i]
                    else:
                        positions[i] = positions[i-1]
                        entry_prices[i] = entry_prices[i-1]
                else:
                    signals[i] = SIZE_HALF
                    positions[i] = positions[i-1] if positions[i-1] != 0 else SIZE_HALF
                    entry_prices[i] = entry_prices[i-1] if entry_prices[i-1] != 0 else close[i]
            else:
                signals[i] = 0.0
                if positions[i-1] != 0:
                    positions[i] = 0.0
                    entry_prices[i] = 0.0
                else:
                    positions[i] = 0.0
        elif trend == -1:  # 4h downtrend
            # Look for MACD histogram crossing below 0 or negative and falling
            if macd_histogram < MACD_SHORT_THRESHOLD:
                if macd_histogram < macd_prev:  # Momentum decreasing
                    signals[i] = -SIZE_FULL
                    if positions[i-1] == 0:
                        positions[i] = -SIZE_FULL
                        entry_prices[i] = close[i]
                    else:
                        positions[i] = positions[i-1]
                        entry_prices[i] = entry_prices[i-1]
                else:
                    signals[i] = -SIZE_HALF
                    positions[i] = positions[i-1] if positions[i-1] != 0 else -SIZE_HALF
                    entry_prices[i] = entry_prices[i-1] if entry_prices[i-1] != 0 else close[i]
            else:
                signals[i] = 0.0
                if positions[i-1] != 0:
                    positions[i] = 0.0
                    entry_prices[i] = 0.0
                else:
                    positions[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
            positions[i] = 0.0
            entry_prices[i] = 0.0
    
    return signals