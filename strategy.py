#!/usr/bin/env python3
"""
EXPERIMENT #007 - Supertrend 6h with Daily EMA Trend Filter
============================================================
Hypothesis: 6h primary timeframe with Supertrend entries + 1d EMA trend filter will outperform
because: (1) Supertrend provides clear trend-following signals with built-in ATR stops,
(2) 6h TF balances signal frequency (less than 4h, more than 12h) reducing fee churn,
(3) Daily EMA(21/55) filter ensures we only trade with major trend,
(4) RSI(14) momentum confirmation avoids entering at extremes,
(5) Conservative position sizing (0.30) controls drawdown.

Key differences from failed attempts:
- Supertrend instead of HMA crossover (proven trend-following indicator)
- 6h primary (middle ground between noisy 4h and slow 12h)
- RSI momentum filter (avoid overbought longs / oversold shorts)
- Proper NaN handling throughout
- Clean position tracking for stoploss

Why this should work:
- 6h timeframe: ~4 bars/day = reasonable signal frequency with low fee impact
- Supertrend(10,3): Adaptive ATR-based trend with built-in volatility adjustment
- Daily EMA filter: Only trade in direction of higher timeframe trend
- RSI filter: Avoid chasing momentum at extremes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_6h_daily_ema_rsi_v1"
timeframe = "6h"
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


def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         period: int = 10, multiplier: float = 3.0) -> tuple:
    """
    Calculate Supertrend indicator.
    Returns: (supertrend_values, supertrend_direction)
    direction: 1 = bullish (price above supertrend), -1 = bearish
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
            
        upper_band = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = -1 if close[i] < supertrend[i] else 1
        else:
            # Update upper/lower bands based on previous supertrend
            if direction[i-1] == 1:
                upper_band = max(upper_band, supertrend[i-1])
                supertrend[i] = lower_band if close[i] < supertrend[i-1] else upper_band
            else:
                lower_band = min(lower_band, supertrend[i-1])
                supertrend[i] = upper_band if close[i] > supertrend[i-1] else lower_band
            
            # Update direction
            direction[i] = 1 if close[i] > supertrend[i] else -1
    
    return supertrend, direction


def calculate_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    return pd.Series(series).ewm(span=span, min_periods=span, adjust=False).mean().values


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
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    ema_1d_fast = calculate_ema(df_1d['close'].values, 21)
    ema_1d_slow = calculate_ema(df_1d['close'].values, 55)
    
    # Align to 6h timeframe with proper shift (Rule 2)
    ema_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_fast)
    ema_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slow)
    
    # Calculate 6h Supertrend for entry signals
    _, supertrend_dir = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Calculate 6h RSI for momentum confirmation
    rsi = calculate_rsi(close, period=14)
    
    # Calculate ATR for additional stoploss reference
    atr = calculate_atr(high, low, close, period=14)
    
    # Initialize signals and tracking variables
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size - conservative for DD control
    
    # Track position for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Minimum period for all calculations
    min_period = 100  # Safe margin for all indicators
    
    for i in range(min_period, n):
        # Skip if any indicator is NaN
        if np.isnan(supertrend_dir[i]):
            continue
        if np.isnan(rsi[i]):
            continue
        if np.isnan(ema_1d_fast_aligned[i]) or np.isnan(ema_1d_slow_aligned[i]):
            continue
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        # Daily trend filter (HTF)
        daily_trend = 0
        if ema_1d_fast_aligned[i] > ema_1d_slow_aligned[i]:
            daily_trend = 1  # Bullish
        elif ema_1d_fast_aligned[i] < ema_1d_slow_aligned[i]:
            daily_trend = -1  # Bearish
        
        # 6h Supertrend direction signal
        st_signal = int(supertrend_dir[i])
        
        # RSI momentum filter (avoid extreme entries)
        rsi_ok_long = rsi[i] < 70  # Not overbought for long entries
        rsi_ok_short = rsi[i] > 30  # Not oversold for short entries
        
        # Only trade in direction of daily trend with RSI confirmation
        target_signal = 0.0
        
        if daily_trend == 1 and st_signal == 1 and rsi_ok_long:
            target_signal = SIZE  # Long
        elif daily_trend == -1 and st_signal == -1 and rsi_ok_short:
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
        elif target_signal < 0 and position_side != -1:
            position_side = -1
            entry_price = close[i]
            lowest_close = close[i]
        elif target_signal == 0 and position_side != 0:
            position_side = 0
        
        signals[i] = target_signal
    
    return signals