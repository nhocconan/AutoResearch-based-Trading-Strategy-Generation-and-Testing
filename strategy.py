#!/usr/bin/env python3
"""
EXPERIMENT #002 - Multi-Timeframe KAMA Trend + MACD Entry + BB Regime Filter
============================================================================
Hypothesis: KAMA adapts to volatility better than HMA/EMA, reducing whipsaw
during sideways markets. MACD histogram provides clearer momentum signals
than RSI for entry timing. Bollinger Band Width filters low-volatility regimes
where trend strategies fail.

Key differences from mtf_hma_rsi_zscore_v1:
- KAMA(48) instead of HMA(48) - adapts to market efficiency ratio
- MACD histogram cross instead of RSI pullback - momentum-based entries
- BB Width regime filter instead of Z-score - detects squeeze/breakout phases
- ATR-based position sizing for consistent risk per trade

Why this might beat Sharpe=1.768:
- KAMA reduces noise in choppy markets (ER-based adaptation)
- MACD histogram captures momentum shifts more precisely than RSI
- BB Width avoids trading during low-volatility consolidation
- Better regime detection = fewer bad trades
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_bbregime_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=48, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    Less lag in trends, more smoothing in noise
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close = np.array(close, dtype=float)
    kama = np.zeros(n)
    
    # Efficiency Ratio: |net change| / sum of absolute changes
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if sum_changes > 0:
            er = net_change / sum_changes
        else:
            er = 0
        
        # Smoothing constant
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    close = pd.Series(close)
    
    ema_fast = close.ewm(span=fast, min_periods=fast).mean().values
    ema_slow = close.ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal_period, min_periods=signal_period).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    atr = np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # EMA-style ATR
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    close = pd.Series(close)
    
    middle = close.rolling(window=period, min_periods=period).mean().values
    std = close.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    band_width = (upper - lower) / middle
    
    return upper, lower, middle, band_width


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal_period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_middle, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate BB Width percentile for regime detection
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.searchsorted(np.sort(x), x.iloc[-1]) / len(x) if len(x) > 0 else 0.5,
        raw=False
    ).values
    
    # 4h KAMA for trend filter (resample 1h → 4h)
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
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(df_4h['close'].values, period=48)
    
    # Calculate 4h trend direction (price vs KAMA + KAMA slope)
    trend_4h = np.zeros(len(kama_4h))
    for i in range(1, len(kama_4h)):
        if kama_4h[i] > 0 and kama_4h[i - 1] > 0:
            price_above = df_4h['close'].values[i] > kama_4h[i]
            kama_rising = kama_4h[i] > kama_4h[i - 1]
            
            if price_above and kama_rising:
                trend_4h[i] = 1  # Bullish
            elif not price_above and not kama_rising:
                trend_4h[i] = -1  # Bearish
            else:
                trend_4h[i] = 0  # Neutral/transition
    
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
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # MACD histogram thresholds
    MACD_THRESHOLD = 0.0  # Histogram must cross zero
    MACD_MIN_CHANGE = 0.001  # Minimum histogram change for confirmation
    
    # BB Width regime thresholds
    BB_WIDTH_LOW = 0.30   # Below 30th percentile = squeeze (avoid)
    BB_WIDTH_HIGH = 0.85  # Above 85th percentile = expansion (good for trends)
    
    # ATR-based volatility filter
    ATR_PCT_MAX = 0.05  # Max 5% ATR relative to price
    
    first_valid = max(48, 26, 20, 14, 100)  # Wait for all indicators
    
    # Track previous signal to detect MACD crosses
    prev_macd_hist = np.zeros(n)
    prev_macd_hist[first_valid:] = macd_hist[first_valid - 1:n - 1]
    
    for i in range(first_valid, n):
        # Check for NaN values
        if (np.isnan(trend_1h[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(bb_width_pct[i]) or np.isnan(atr_1h[i])):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        hist = macd_hist[i]
        prev_hist = prev_macd_hist[i]
        bb_pct = bb_width_pct[i]
        atr_pct = atr_1h[i] / close[i] if close[i] > 0 else 1.0
        
        # Volatility filter - don't trade during extreme ATR
        if atr_pct > ATR_PCT_MAX:
            signals[i] = 0.0
            continue
        
        # BB regime filter - avoid squeeze zones, prefer expansion
        if bb_pct < BB_WIDTH_LOW:
            # Squeeze zone - reduce position size or stay out
            regime_multiplier = 0.5
        elif bb_pct > BB_WIDTH_HIGH:
            # Expansion zone - full position allowed
            regime_multiplier = 1.0
        else:
            # Normal zone
            regime_multiplier = 0.75
        
        # MACD momentum confirmation
        macd_bullish = hist > MACD_THRESHOLD and hist > prev_hist
        macd_bearish = hist < MACD_THRESHOLD and hist < prev_hist
        macd_strong_bullish = hist > MACD_THRESHOLD and prev_hist <= MACD_THRESHOLD
        macd_strong_bearish = hist < MACD_THRESHOLD and prev_hist >= MACD_THRESHOLD
        
        if trend == 1:  # 4h uptrend
            if macd_strong_bullish:
                # Fresh bullish cross - full position
                signals[i] = SIZE_FULL * regime_multiplier
            elif macd_bullish:
                # Continued bullish momentum - half position
                signals[i] = SIZE_HALF * regime_multiplier
            else:
                # No momentum - exit
                signals[i] = 0.0
        elif trend == -1:  # 4h downtrend
            if macd_strong_bearish:
                # Fresh bearish cross - full short
                signals[i] = -SIZE_FULL * regime_multiplier
            elif macd_bearish:
                # Continued bearish momentum - half short
                signals[i] = -SIZE_HALF * regime_multiplier
            else:
                # No momentum - exit
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    # Apply ATR-based trailing stoploss logic
    # If signal changes sign or goes to 0, that's the stoploss
    # Additional: if price moves 2*ATR against position, force signal to 0
    
    # Calculate running max/min for trailing stop
    running_max = pd.Series(close).rolling(window=20, min_periods=1).max().values
    running_min = pd.Series(close).rolling(window=20, min_periods=1).min().values
    
    for i in range(first_valid, n):
        if signals[i] > 0:  # Long position
            # Check if price dropped 2*ATR from recent high
            stop_level = running_max[i] - 2 * atr_1h[i]
            if close[i] < stop_level and signals[i] > 0:
                signals[i] = 0.0  # Stoploss triggered
        elif signals[i] < 0:  # Short position
            # Check if price rose 2*ATR from recent low
            stop_level = running_min[i] + 2 * atr_1h[i]
            if close[i] > stop_level and signals[i] < 0:
                signals[i] = 0.0  # Stoploss triggered
    
    # Round to discrete levels to minimize churn
    for i in range(n):
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif abs(signals[i]) < 0.27:
            signals[i] = 0.20 * np.sign(signals[i])
        else:
            signals[i] = 0.35 * np.sign(signals[i])
    
    return signals