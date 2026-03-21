#!/usr/bin/env python3
"""
EXPERIMENT #006 - HMA Crossover 12h with Daily Trend Filter
============================================================
Hypothesis: 12h primary timeframe with 1d HTF trend filter will outperform 4h strategies
because: (1) fewer signals = less fee churn, (2) HMA is more responsive than EMA for
trend changes, (3) daily trend filter prevents counter-trend trades in strong regimes,
(4) ATR trailing stop protects against major drawdowns.

Key differences from failed attempts:
- NO RSI pullback (failed in #001, #002, #005)
- 12h primary instead of 4h (higher TF = less noise, less fees)
- HMA crossover instead of Supertrend or simple EMA
- 1d HTF filter (not 4h or 6h which may be too noisy)
- Proper ATR trailing stop for risk management
- Conservative position sizing (0.30) to control DD

Why this should work:
- 12h timeframe: ~2 bars/day vs 96 bars/day on 15m = 48x less fee impact
- HMA(16/48): Faster response than EMA while filtering noise
- Daily trend filter: Only trade in direction of higher timeframe trend
- ATR stop: Dynamic risk management adapts to volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_12h_daily_trend_atr_v1"
timeframe = "12h"
leverage = 1.0


def calculate_hma(series: np.ndarray, period: int) -> np.ndarray:
    """Calculate Hull Moving Array using WMA formula."""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(arr, w):
        result = np.full(len(arr), np.nan)
        weights = np.arange(1, w + 1)
        for i in range(w - 1, len(arr)):
            if np.any(np.isnan(arr[i-w+1:i+1])):
                continue
            result[i] = np.dot(arr[i-w+1:i+1], weights) / weights.sum()
        return result
    
    half = period // 2
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n))
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, int(np.sqrt(period)))
    
    return hma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily HMA for trend filter
    hma_1d_fast = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slow = calculate_hma(df_1d['close'].values, 55)
    
    # Align to 12h timeframe with proper shift (Rule 2)
    hma_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_fast)
    hma_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slow)
    
    # Calculate 12h HMA for entry signals
    hma_12h_fast = calculate_hma(close, 16)
    hma_12h_slow = calculate_hma(close, 48)
    
    # Calculate ATR for trailing stop
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Initialize signals and tracking variables
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size - conservative for DD control
    
    # Track position for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Minimum period for HMA calculations
    min_period = 100  # Safe margin for all indicators
    
    for i in range(min_period, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_12h_fast[i]) or np.isnan(hma_12h_slow[i]):
            continue
        if np.isnan(hma_1d_fast_aligned[i]) or np.isnan(hma_1d_slow_aligned[i]):
            continue
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        # Daily trend filter (HTF)
        daily_trend = 0
        if hma_1d_fast_aligned[i] > hma_1d_slow_aligned[i]:
            daily_trend = 1  # Bullish
        elif hma_1d_fast_aligned[i] < hma_1d_slow_aligned[i]:
            daily_trend = -1  # Bearish
        
        # 12h HMA crossover signal
        hma_signal = 0
        if hma_12h_fast[i] > hma_12h_slow[i]:
            hma_signal = 1  # Bullish crossover
        elif hma_12h_fast[i] < hma_12h_slow[i]:
            hma_signal = -1  # Bearish crossover
        
        # Only trade in direction of daily trend
        if daily_trend == 1 and hma_signal == 1:
            target_signal = SIZE
        elif daily_trend == -1 and hma_signal == -1:
            target_signal = -SIZE
        else:
            target_signal = 0.0
        
        # ATR trailing stop logic (Rule 6)
        current_signal = signals[i-1] if i > 0 else 0.0
        
        if position_side == 1:  # Long position
            highest_close = max(highest_close, close[i])
            stop_price = highest_close - 2.5 * atr[i]
            if close[i] < stop_price:
                target_signal = 0.0  # Stoploss hit
                position_side = 0
        elif position_side == -1:  # Short position
            lowest_close = min(lowest_close, close[i])
            stop_price = lowest_close + 2.5 * atr[i]
            if close[i] > stop_price:
                target_signal = 0.0  # Stoploss hit
                position_side = 0
        
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