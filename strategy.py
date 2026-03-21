#!/usr/bin/env python3
"""
EXPERIMENT #003 - Supertrend(6h) + ADX Trend Strength + Volume Filter
======================================================================
Hypothesis: 6h Supertrend captures major trends with fewer whipsaws than 4h.
Adding ADX(14)>25 filter avoids choppy sideways markets where Supertrend fails.
Volume confirmation ensures breakout legitimacy. Explicit ATR stoploss limits DD.

Why this differs from failed strategies:
- Supertrend (ATR-based) vs HMA/EMA (price-based) - adapts to volatility
- ADX filter removes low-trend regimes (where previous strategies whipsawed)
- Volume confirmation adds conviction filter
- Explicit stoploss logic (2*ATR) vs implicit trend-following exit
- 6h timeframe = cleaner than 4h, more signals than 12h

Key risk controls:
- Signal magnitude: 0.30 (30% position size, not 100%)
- Stoploss: signal→0 when price moves 2*ATR against position
- Discrete levels: 0.0, ±0.30 to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_6h_adx_volume_v1"
timeframe = "6h"
leverage = 1.0


def calculate_atr(high, low, close, period=10):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_adx(high, low, close, period=14):
    """Calculate ADX(14) for trend strength"""
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's method (EMA with alpha=1/period)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    atr = np.zeros(n)
    
    # Initialize first period sum
    sum_tr = tr[:period].sum()
    sum_plus_dm = plus_dm[:period].sum()
    sum_minus_dm = minus_dm[:period].sum()
    
    atr[period-1] = sum_tr / period
    plus_di[period-1] = 100 * (sum_plus_dm / sum_tr) if sum_tr > 0 else 0
    minus_di[period-1] = 100 * (sum_minus_dm / sum_tr) if sum_tr > 0 else 0
    
    # Wilder's smoothing for remaining bars
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        sum_plus_dm = plus_dm[i] + (plus_di[i-1] / 100) * atr[i] * (period - 1) / 100
        sum_minus_dm = minus_dm[i] + (minus_di[i-1] / 100) * atr[i] * (period - 1) / 100
        
        # Recalculate DI from smoothed DM
        if atr[i] > 0:
            plus_di[i] = 100 * (plus_dm[i] + (plus_di[i-1] / 100) * atr[i] * (period - 1) / 100) / atr[i]
            minus_di[i] = 100 * (minus_dm[i] + (minus_di[i-1] / 100) * atr[i] * (period - 1) / 100) / atr[i]
        else:
            plus_di[i] = plus_di[i-1]
            minus_di[i] = minus_di[i-1]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # ADX = SMA of DX
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Calculate Supertrend with proper state tracking"""
    n = len(close)
    supertrend = np.zeros(n)
    trend_direction = np.zeros(n)  # 1 = long, -1 = short
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Find first valid ATR index
    first_valid = 0
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            first_valid = i
            break
    
    if first_valid >= n - 1:
        return supertrend, trend_direction
    
    # Initialize
    supertrend[first_valid] = upper_band[first_valid]
    trend_direction[first_valid] = -1  # Start with short bias
    
    for i in range(first_valid + 1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1]
            trend_direction[i] = trend_direction[i-1]
            continue
        
        if trend_direction[i-1] == 1:
            # Previous trend was long
            if close[i] > supertrend[i-1]:
                # Stay long, use lower band (trailing)
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                trend_direction[i] = 1
            else:
                # Flip to short
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
        else:
            # Previous trend was short
            if close[i] < supertrend[i-1]:
                # Stay short, use upper band (trailing)
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                trend_direction[i] = -1
            else:
                # Flip to long
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
    
    return supertrend, trend_direction


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (for regime filter)
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    daily_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate indicators on primary 6h timeframe
    atr = calculate_atr(high, low, close, period=10)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume SMA(20) for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily SMA(50) for long-term trend regime
    daily_sma50 = pd.Series(daily_close).rolling(window=50, min_periods=50).mean().values
    daily_sma50_aligned = align_htf_to_ltf(prices, df_1d, daily_sma50)
    
    # Calculate Supertrend
    supertrend, trend_direction = calculate_supertrend(high, low, close, atr, multiplier=3.0)
    
    # Generate signals with discrete position sizing and stoploss
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size - conservative for drawdown control
    ADX_THRESHOLD = 25  # Minimum trend strength
    ATR_STOP_MULT = 2.0  # Stoploss at 2*ATR against position
    
    # Track entry prices for stoploss
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    
    # Find first valid index (all indicators ready)
    first_valid = max(50, 20)  # ATR(10), ADX(14), Volume SMA(20), Daily SMA(50)
    
    for i in range(first_valid, n):
        # Check for daily trend regime (only trade in direction of daily trend)
        daily_trend_bullish = daily_aligned[i] > daily_sma50_aligned[i] if not np.isnan(daily_sma50_aligned[i]) else True
        daily_trend_bearish = daily_aligned[i] < daily_sma50_aligned[i] if not np.isnan(daily_sma50_aligned[i]) else False
        
        # Volume confirmation
        volume_confirmed = volume[i] > volume_sma[i] if not np.isnan(volume_sma[i]) else True
        
        # ADX trend strength filter
        trend_strong = not np.isnan(adx[i]) and adx[i] > ADX_THRESHOLD
        
        # Supertrend signal
        supertrend_long = trend_direction[i] == 1
        supertrend_short = trend_direction[i] == -1
        
        # Check stoploss first (before new signals)
        if position_side == 1 and entry_price > 0:
            # Long position - check if price dropped 2*ATR below entry
            if close[i] < entry_price - ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
        
        if position_side == -1 and entry_price > 0:
            # Short position - check if price rose 2*ATR above entry
            if close[i] > entry_price + ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
        
        # Generate new signals (only if flat or trend confirms)
        if supertrend_long and trend_strong and volume_confirmed:
            # Only go long if daily trend is bullish or neutral
            if daily_trend_bullish or np.isnan(daily_sma50_aligned[i]):
                signals[i] = SIZE
                position_side = 1
                entry_price = close[i]
            elif position_side == -1:
                # Close short and go flat
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        
        elif supertrend_short and trend_strong and volume_confirmed:
            # Only go short if daily trend is bearish or neutral
            if daily_trend_bearish or np.isnan(daily_sma50_aligned[i]):
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
            elif position_side == 1:
                # Close long and go flat
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        
        else:
            # No clear signal - maintain or flatten
            if not trend_strong:
                # ADX too low - flatten to avoid chop
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals