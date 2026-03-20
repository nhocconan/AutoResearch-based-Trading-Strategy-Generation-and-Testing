#!/usr/bin/env python3
"""
EXPERIMENT #011 - KAMA Adaptive Trend + MACD Momentum + Z-Score Filter
=======================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility
better than static MAs, reducing whipsaw in choppy markets. Combined with MACD
histogram for momentum entry timing and Z-score for regime filtering, this should
improve risk-adjusted returns over Donchian breakouts.

Key differences from mtf_donchian_rsi_atr_v1:
- KAMA(10,2,30) instead of Donchian - adapts to market efficiency ratio
- MACD histogram crosses for entry timing instead of RSI pullbacks
- Z-score(20) filter to avoid trading at price extremes
- Multi-timeframe: 4h KAMA trend + 1h MACD entries

Why this might beat Sharpe=2.931:
- KAMA reduces lag in trending markets, flattens in chop
- MACD histogram provides cleaner momentum signals than RSI
- Z-score filter avoids buying tops/selling bottoms
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_zscore_v2"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise via Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period - 1, n):
        net_change = abs(close[i] - close[i - er_period + 1])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(er_period - 1, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period - 1] = close[er_period - 1]
    
    # Calculate KAMA
    for i in range(er_period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    # EMA calculation helper
    def ema(series, period):
        result = np.zeros(n)
        multiplier = 2.0 / (period + 1)
        result[period - 1] = np.mean(series[:period])
        for i in range(period, n):
            result[i] = (series[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = np.zeros(n)
    multiplier = 2.0 / (signal + 1)
    first_valid = fast + signal - 1
    if first_valid < n:
        signal_line[first_valid] = np.mean(macd_line[fast:first_valid + 1])
        for i in range(first_valid + 1, n):
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
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
    n_4h = len(c_4h)
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(n_4h)
    for i in range(30, n_4h):
        if kama_4h[i] > 0:
            # KAMA slope
            kama_slope = kama_4h[i] - kama_4h[i - 5] if i >= 5 else 0
            # Price position relative to KAMA
            price_ratio = (c_4h[i] - kama_4h[i]) / kama_4h[i] if kama_4h[i] > 0 else 0
            
            if kama_slope > 0 and price_ratio > -0.02:
                trend_4h[i] = 1  # Bullish
            elif kama_slope < 0 and price_ratio < 0.02:
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
    
    # MACD histogram thresholds for momentum entry
    MACD_LONG_THRESH = 0.0    # Histogram crossing above zero
    MACD_SHORT_THRESH = 0.0   # Histogram crossing below zero
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 1.5    # Don't buy when price > 1.5 std above mean
    ZSCORE_MIN = -1.5   # Don't sell when price < 1.5 std below mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(30, 35, 20, 14)  # Wait for all indicators
    
    # Track positions for trailing stop logic
    position_long = False
    position_short = False
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(macd_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_hist = hist_1h[i]
        macd_hist_prev = hist_1h[i - 1] if i > 0 else 0
        zscore = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Skip if ATR is extremely high (>5% of price)
        if atr > 0 and atr / price > 0.05:
            signals[i] = 0.0
            position_long = False
            position_short = False
            continue
        
        # Z-score filter - avoid extremes
        if zscore > ZSCORE_MAX or zscore < ZSCORE_MIN:
            signals[i] = 0.0
            position_long = False
            position_short = False
            continue
        
        # Check trailing stop for existing positions
        if position_long:
            highest_since_entry = max(highest_since_entry, price)
            stoploss_price = highest_since_entry - ATR_STOP_MULT * atr
            
            if price < stoploss_price:
                # Stoploss triggered
                signals[i] = 0.0
                position_long = False
                entry_price = 0.0
                highest_since_entry = 0.0
            elif macd_hist < MACD_LONG_THRESH and macd_hist_prev >= MACD_LONG_THRESH:
                # MACD histogram crossed below zero - exit long
                signals[i] = 0.0
                position_long = False
                entry_price = 0.0
                highest_since_entry = 0.0
            else:
                signals[i] = SIZE_FULL  # Hold long position
            continue
        
        if position_short:
            lowest_since_entry = min(lowest_since_entry, price)
            stoploss_price = lowest_since_entry + ATR_STOP_MULT * atr
            
            if price > stoploss_price:
                # Stoploss triggered
                signals[i] = 0.0
                position_short = False
                entry_price = 0.0
                lowest_since_entry = 0.0
            elif macd_hist > MACD_SHORT_THRESH and macd_hist_prev <= MACD_SHORT_THRESH:
                # MACD histogram crossed above zero - exit short
                signals[i] = 0.0
                position_short = False
                entry_price = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -SIZE_FULL  # Hold short position
            continue
        
        # Entry logic
        if trend == 1:  # 4h uptrend
            if macd_hist > MACD_LONG_THRESH and macd_hist_prev <= MACD_LONG_THRESH:
                # MACD histogram crossed above zero - enter long
                signals[i] = SIZE_FULL
                position_long = True
                entry_price = price
                highest_since_entry = price
            elif macd_hist > 0:
                # MACD positive but no cross - half position
                signals[i] = SIZE_HALF
            else:
                signals[i] = 0.0
        elif trend == -1:  # 4h downtrend
            if macd_hist < MACD_SHORT_THRESH and macd_hist_prev >= MACD_SHORT_THRESH:
                # MACD histogram crossed below zero - enter short
                signals[i] = -SIZE_FULL
                position_short = True
                entry_price = price
                lowest_since_entry = price
            elif macd_hist < 0:
                # MACD negative but no cross - half position
                signals[i] = -SIZE_HALF
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals