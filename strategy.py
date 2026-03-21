#!/usr/bin/env python3
"""
EXPERIMENT #008 - KAMA 12h with Daily EMA Trend + BBW Regime Filter
====================================================================
Hypothesis: 12h primary timeframe with KAMA adaptive trend + Daily EMA filter + 
Bollinger Band Width regime detection will outperform because:

1. KAMA (Kaufman Adaptive Moving Average) adapts to market noise - faster in trends, 
   slower in chop - superior to static EMA/HMA for crypto's varying volatility
2. 12h timeframe: ~2 bars/day = minimal fee churn, cleaner signals than 4h/6h
3. Daily EMA(21/55) filter ensures we only trade with major trend direction
4. BBW (Bollinger Band Width) percentile filter avoids trading during squeeze/low-vol 
   periods where trend strategies fail
5. Conservative position sizing (0.25) with ATR stoploss controls drawdown

Key differences from failed attempts:
- KAMA instead of Supertrend/HMA/EMA (adaptive to volatility regimes)
- 12h primary (higher than failed 6h attempts = less noise)
- BBW regime filter (NEW - avoids low-volatility chop)
- Simpler position tracking (no complex state machine that may crash)
- More robust NaN handling throughout

Why this should work:
- KAMA's Efficiency Ratio adapts to market conditions automatically
- 12h TF: ~2 signals/day = low fee impact, strong trend capture
- BBW filter: Only trade when volatility is expanding (trending markets)
- Daily trend filter: Proven concept from baseline strategies
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_12h_daily_ema_bbw_v1"
timeframe = "12h"
leverage = 1.0


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range."""
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


def calculate_kama(close: np.ndarray, er_period: int = 10, fast_sc: int = 2, slow_sc: int = 30) -> np.ndarray:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + 1:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc_val = 2.0 / (fast_sc + 1)
    slow_sc_val = 2.0 / (slow_sc + 1)
    sc = er * (fast_sc_val - slow_sc_val) + slow_sc_val
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


def calculate_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    return pd.Series(series).ewm(span=span, min_periods=span, adjust=False).mean().values


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> tuple:
    """Calculate Bollinger Bands and Band Width."""
    n = len(close)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bbw = (upper - lower) / sma  # Band Width as % of price
    
    return upper, lower, bbw


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index."""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    # Extract price data
    close = prices["close"].values.astype(float)
    high = prices["high"].values.astype(float)
    low = prices["low"].values.astype(float)
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    ema_1d_fast = calculate_ema(df_1d['close'].values.astype(float), 21)
    ema_1d_slow = calculate_ema(df_1d['close'].values.astype(float), 55)
    
    # Align to 12h timeframe with proper shift (Rule 2)
    ema_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_fast)
    ema_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slow)
    
    # Calculate 12h KAMA for adaptive trend following
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Calculate 12h Bollinger Band Width for regime filter
    _, _, bbw = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate BBW percentile for regime detection (rolling 100 bars)
    bbw_percentile = np.full(n, np.nan)
    for i in range(100, n):
        if not np.isnan(bbw[i]):
            bbw_window = bbw[i-100:i+1]
            bbw_window = bbw_window[~np.isnan(bbw_window)]
            if len(bbw_window) > 0:
                bbw_percentile[i] = np.sum(bbw_window < bbw[i]) / len(bbw_window)
    
    # Calculate 12h RSI for momentum confirmation
    rsi = calculate_rsi(close, period=14)
    
    # Calculate ATR for stoploss
    atr = calculate_atr(high, low, close, period=14)
    
    # Initialize signals
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - conservative for DD control
    
    # Track position for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Minimum period for all calculations
    min_period = 150  # Safe margin for all indicators
    
    for i in range(min_period, n):
        # Skip if any indicator is NaN
        if np.isnan(kama[i]):
            continue
        if np.isnan(rsi[i]):
            continue
        if np.isnan(ema_1d_fast_aligned[i]) or np.isnan(ema_1d_slow_aligned[i]):
            continue
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        if np.isnan(bbw_percentile[i]):
            continue
        
        # Daily trend filter (HTF)
        daily_trend = 0
        if ema_1d_fast_aligned[i] > ema_1d_slow_aligned[i]:
            daily_trend = 1  # Bullish
        elif ema_1d_fast_aligned[i] < ema_1d_slow_aligned[i]:
            daily_trend = -1  # Bearish
        
        # KAMA trend direction (price relative to KAMA)
        kama_signal = 1 if close[i] > kama[i] else -1
        
        # BBW regime filter - only trade when volatility is expanding (percentile > 0.5)
        # Avoid squeeze periods where trend strategies fail
        bbw_ok = bbw_percentile[i] > 0.4  # At least 40th percentile
        
        # RSI momentum filter (avoid extreme entries)
        rsi_ok_long = rsi[i] < 75  # Not extremely overbought for long entries
        rsi_ok_short = rsi[i] > 25  # Not extremely oversold for short entries
        
        # Determine target signal
        target_signal = 0.0
        
        if daily_trend == 1 and kama_signal == 1 and bbw_ok and rsi_ok_long:
            target_signal = SIZE  # Long
        elif daily_trend == -1 and kama_signal == -1 and bbw_ok and rsi_ok_short:
            target_signal = -SIZE  # Short
        else:
            target_signal = 0.0  # Flat
        
        # ATR trailing stop logic (Rule 6)
        if position_side == 1:  # Long position
            highest_close = max(highest_close, close[i])
            stop_price = highest_close - 2.5 * atr[i]
            if close[i] < stop_price:
                target_signal = 0.0  # Stoploss hit
        elif position_side == -1:  # Short position
            lowest_close = min(lowest_close, close[i])
            stop_price = lowest_close + 2.5 * atr[i]
            if close[i] > stop_price:
                target_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        if target_signal > 0 and position_side != 1:
            position_side = 1
            entry_price = close[i]
            highest_close = close[i]
            lowest_close = close[i]
        elif target_signal < 0 and position_side != -1:
            position_side = -1
            entry_price = close[i]
            highest_close = close[i]
            lowest_close = close[i]
        elif target_signal == 0 and position_side != 0:
            position_side = 0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = target_signal
    
    return signals