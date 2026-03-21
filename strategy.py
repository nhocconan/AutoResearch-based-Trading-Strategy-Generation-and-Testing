#!/usr/bin/env python3
"""
EXPERIMENT #010 - Supertrend with Daily Filter and RSI Pullback (4h)
=====================================================================
Hypothesis: Supertrend(10,3) identifies trend direction on 4h, Daily HMA(50)
filters for major trend alignment, and RSI(14) pullback entries avoid chasing
breakouts. This combination should produce more trades than pure crossover
strategies while maintaining trend quality.

Key features:
- Primary TF: 4h (4-hour candles)
- HTF filter: 1d HMA(50) for major trend direction
- Entry: Supertrend flip + RSI pullback (40-60 range for fresh entries)
- Filter: Only trade when 4h supertrend aligns with daily HMA trend
- Stoploss: Supertrend level (built-in trailing stop)
- Position sizing: 0.25 discrete levels, max 0.30
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_daily_rsi_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing method"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if direction[i-1] == 1:
            if close[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            if close[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    return supertrend, direction


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    
    avg_gain[period] = gain.iloc[:period+1].mean()
    avg_loss[period] = loss.iloc[:period+1].mean()
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30  # Maximum position size
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    min_period = 80  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Skip if any indicator is NaN
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter - only trade in direction of daily HMA
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # 4h Supertrend direction
        st_trend = int(st_direction[i])
        
        # RSI pullback filter - enter on pullback, not at extremes
        # For longs: RSI should be 35-55 (pullback in uptrend)
        # For shorts: RSI should be 45-65 (pullback in downtrend)
        rsi_long_valid = 35 < rsi[i] < 55
        rsi_short_valid = 45 < rsi[i] < 65
        
        # Determine target signal
        target_signal = 0.0
        
        # Long entry: Supertrend bullish + Daily trend bullish + RSI pullback
        if st_trend == 1 and daily_trend == 1 and rsi_long_valid:
            if position_side <= 0:  # Flat or short - new long entry
                target_signal = SIZE
            elif position_side == 1:  # Already long - maintain
                target_signal = SIZE
        
        # Short entry: Supertrend bearish + Daily trend bearish + RSI pullback
        elif st_trend == -1 and daily_trend == -1 and rsi_short_valid:
            if position_side >= 0:  # Flat or long - new short entry
                target_signal = -SIZE
            elif position_side == -1:  # Already short - maintain
                target_signal = -SIZE
        
        # Exit signal: Trend reversal detected
        if position_side == 1 and st_trend == -1:
            target_signal = 0.0
        elif position_side == -1 and st_trend == 1:
            target_signal = 0.0
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                # Trailing stop using Supertrend level
                trailing_stop = supertrend[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                # Trailing stop using Supertrend level
                trailing_stop = supertrend[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals