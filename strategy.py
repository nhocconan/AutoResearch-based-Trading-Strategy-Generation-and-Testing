#!/usr/bin/env python3
"""
EXPERIMENT #009 - Donchian 4h Breakout with Daily Trend + Volume Confirm
=========================================================================
Hypothesis: 4h Donchian breakout strategy with Daily EMA trend filter and 
volume confirmation will outperform because:

1. Donchian breakouts (20-period high/low) are proven trend-following signals
   that capture sustained moves without whipsaw from mean-reversion indicators
2. 4h timeframe: ~6 bars/day = good balance of signal frequency vs noise
3. Daily EMA(21/55) filter ensures we only trade with major trend direction
4. Volume confirmation filters out false breakouts (breakout volume > 1.5x avg)
5. ATR trailing stop (2.5x) protects capital during reversals
6. Conservative position sizing (0.30) with discrete levels minimizes fee churn

Key differences from failed attempts:
- Donchian breakout instead of KAMA/Supertrend/HMA (pure price action)
- Volume confirmation on entries (NEW - filters false breakouts)
- 4h primary (lower than crashed 12h/6h attempts = more signals, proven TF)
- Simpler logic without complex regime/BBW filters that may crash
- Robust NaN handling and column access

Why this should work:
- Donchian channels are time-tested (Turtle Trading system)
- Volume confirms genuine breakout interest vs fake moves
- Daily trend filter prevents counter-trend trades
- 4h TF: proven working timeframe in crypto (less noise than 1h, more signals than 12h)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_4h_daily_volume_v1"
timeframe = "4h"
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


def calculate_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    return pd.Series(series).ewm(span=span, min_periods=span, adjust=False).mean().values


def calculate_donchian(high: np.ndarray, low: np.ndarray, period: int = 20) -> tuple:
    """
    Calculate Donchian Channel (highest high / lowest low over period).
    Returns: (upper_band, lower_band)
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower


def calculate_volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate simple moving average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    # Extract price data - use standard column names
    close = prices["close"].values.astype(float)
    high = prices["high"].values.astype(float)
    low = prices["low"].values.astype(float)
    volume = prices["volume"].values.astype(float)
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    ema_1d_fast = calculate_ema(df_1d['close'].values.astype(float), 21)
    ema_1d_slow = calculate_ema(df_1d['close'].values.astype(float), 55)
    
    # Align to 4h timeframe with proper shift (Rule 2)
    ema_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_fast)
    ema_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slow)
    
    # Calculate 4h Donchian Channel (20-period breakout)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 4h volume SMA for confirmation
    volume_sma = calculate_volume_sma(volume, period=20)
    
    # Calculate 4h ATR for stoploss
    atr = calculate_atr(high, low, close, period=14)
    
    # Initialize signals
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(ema_1d_fast_aligned[i]) or np.isnan(ema_1d_slow_aligned[i]):
            continue
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # Daily trend filter (HTF)
        daily_trend = 0
        if ema_1d_fast_aligned[i] > ema_1d_slow_aligned[i]:
            daily_trend = 1  # Bullish
        elif ema_1d_fast_aligned[i] < ema_1d_slow_aligned[i]:
            daily_trend = -1  # Bearish
        
        # Donchian breakout detection
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # Volume confirmation (current volume > 1.5x average)
        volume_confirmed = volume[i] > 1.5 * volume_sma[i]
        
        # Determine target signal
        target_signal = 0.0
        
        if daily_trend == 1 and breakout_long and volume_confirmed:
            target_signal = SIZE  # Long breakout with trend + volume
        elif daily_trend == -1 and breakout_short and volume_confirmed:
            target_signal = -SIZE  # Short breakout with trend + volume
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