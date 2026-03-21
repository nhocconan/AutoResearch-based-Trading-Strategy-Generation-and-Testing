#!/usr/bin/env python3
"""
EXPERIMENT #013 - KAMA Trend + MACD Momentum + Z-Score Regime Filter
======================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA,
providing cleaner trend signals in choppy markets. Combined with MACD histogram for
momentum confirmation and Z-score for regime detection, this should reduce false entries
while maintaining trend exposure. ATR trailing stops protect against reversals.

Key differences from mtf_hma_supertrend_rsi_v1:
- KAMA(10,2,30) instead of HMA for trend (adaptive to market efficiency)
- MACD histogram cross for entry timing (momentum confirmation vs RSI pullback)
- Z-score(20) filter to avoid extreme deviations (mean reversion risk)
- Multi-timeframe: 4h KAMA trend + 1h MACD entries

Why this might beat Sharpe=2.931:
- KAMA reduces whipsaw in ranging markets better than HMA
- MACD histogram provides earlier momentum signals than RSI
- Z-score filter avoids entering at extreme extensions
- ATR stops adapt to volatility regime changes
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_zscore_v3"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market efficiency - moves fast in trends, slow in ranges
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    # EMA calculation helper
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
    signal_line[first_valid] = np.mean(macd_line[fast:first_valid + 1])
    for i in range(first_valid + 1, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score for regime detection"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period, n):
        window = close[i - period:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore[i] = (close[i] - mean) / std
    
    return zscore


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    macd_1h, signal_1h, hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    zscore_1h = calculate_zscore(close, period=20)
    atr_1h = calculate_atr(high, low, close, period=14)
    
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
    
    c_4h = df_4h['close'].values
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(40, len(c_4h)):
        if kama_4h[i] > 0:
            # Price above KAMA + KAMA sloping up = bullish
            kama_slope = kama_4h[i] - kama_4h[i - 5]
            price_above_kama = c_4h[i] > kama_4h[i]
            
            if price_above_kama and kama_slope > 0:
                trend_4h[i] = 1  # Bullish
            elif not price_above_kama and kama_slope < 0:
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
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # MACD histogram thresholds for entry confirmation
    MACD_HIST_THRESHOLD = 0.0  # Histogram must cross above/below zero
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 2.0     # Don't enter if price is > 2 std from mean
    ZSCORE_MIN = -2.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 40, 26, 20, 14)  # Wait for all indicators
    
    # Track entry prices and stops for trailing stop logic
    entry_prices = np.zeros(n)
    stop_prices = np.zeros(n)
    position_direction = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    
    for i in range(first_valid, n):
        if np.isnan(macd_1h[i]) or np.isnan(hist_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_hist = hist_1h[i]
        macd_hist_prev = hist_1h[i - 1] if i > 0 else 0
        zscore = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Z-score filter - avoid extreme deviations (mean reversion risk)
        if zscore > ZSCORE_MAX or zscore < ZSCORE_MIN:
            # If we have an existing position, check stoploss
            if position_direction[i - 1] != 0 and i > 0:
                if position_direction[i - 1] == 1:
                    stop_price = stop_prices[i - 1]
                    if price < stop_price:
                        signals[i] = 0.0
                        position_direction[i] = 0
                    else:
                        signals[i] = signals[i - 1]
                        position_direction[i] = position_direction[i - 1]
                elif position_direction[i - 1] == -1:
                    stop_price = stop_prices[i - 1]
                    if price > stop_price:
                        signals[i] = 0.0
                        position_direction[i] = 0
                    else:
                        signals[i] = signals[i - 1]
                        position_direction[i] = position_direction[i - 1]
            else:
                signals[i] = 0.0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            continue
        
        if trend == 1:  # 4h uptrend - look for long entries
            # MACD histogram cross above zero (momentum confirmation)
            if macd_hist > MACD_HIST_THRESHOLD and macd_hist_prev <= MACD_HIST_THRESHOLD:
                # Fresh long entry
                signals[i] = SIZE_FULL
                entry_prices[i] = price
                stop_prices[i] = price - ATR_STOP_MULT * atr
                position_direction[i] = 1
            elif signals[i - 1] > 0 and i > 0:
                # Hold existing long position with trailing stop
                prev_stop = stop_prices[i - 1]
                new_stop = price - ATR_STOP_MULT * atr
                
                # Trail stop up (never down)
                trailing_stop = max(prev_stop, new_stop)
                stop_prices[i] = trailing_stop
                
                if price < trailing_stop:
                    signals[i] = 0.0  # Stoploss triggered
                    position_direction[i] = 0
                else:
                    signals[i] = signals[i - 1]  # Hold position
                    position_direction[i] = 1
            else:
                signals[i] = 0.0
                position_direction[i] = 0
                
        elif trend == -1:  # 4h downtrend - look for short entries
            # MACD histogram cross below zero (momentum confirmation)
            if macd_hist < -MACD_HIST_THRESHOLD and macd_hist_prev >= -MACD_HIST_THRESHOLD:
                # Fresh short entry
                signals[i] = -SIZE_FULL
                entry_prices[i] = price
                stop_prices[i] = price + ATR_STOP_MULT * atr
                position_direction[i] = -1
            elif signals[i - 1] < 0 and i > 0:
                # Hold existing short position with trailing stop
                prev_stop = stop_prices[i - 1]
                new_stop = price + ATR_STOP_MULT * atr
                
                # Trail stop down (never up)
                trailing_stop = min(prev_stop, new_stop)
                stop_prices[i] = trailing_stop
                
                if price > trailing_stop:
                    signals[i] = 0.0  # Stoploss triggered
                    position_direction[i] = 0
                else:
                    signals[i] = signals[i - 1]  # Hold position
                    position_direction[i] = -1
            else:
                signals[i] = 0.0
                position_direction[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_direction[i] = 0
    
    return signals