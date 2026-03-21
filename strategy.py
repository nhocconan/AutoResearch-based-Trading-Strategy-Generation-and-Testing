#!/usr/bin/env python3
"""
EXPERIMENT #004 - Supertrend + Daily Trend Filter (4h Primary)
=================================================================
Hypothesis: 4h Supertrend captures medium-term trends while daily HMA(50)
filter ensures we trade only in direction of long-term trend. RSI filter
avoids entering at extremes. ATR trailing stop protects capital.

Key features:
- Primary TF: 4h (4-hour candles)
- HTF filter: 1d HMA(50) for major trend direction
- Entry: Supertrend(10, 3) flip signals
- Filter: RSI(14) between 40-60 (not overbought/oversold)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels

Why this should work:
- 4h provides enough signals (vs 1d which is too slow)
- Daily trend filter reduces whipsaws in choppy markets
- Supertrend adapts to volatility via ATR
- Discrete sizing minimizes fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_daily_filter_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            direction[i] = 0
            continue
        
        mid = (high[i] + low[i]) / 2
        upper_band = mid + multiplier * atr[i]
        lower_band = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            # Update upper/lower bands based on previous direction
            if direction[i-1] == 1:
                upper_band = min(upper_band, supertrend[i-1])
                if close[i] > supertrend[i-1]:
                    supertrend[i] = upper_band
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band
                    direction[i] = -1
            else:
                lower_band = max(lower_band, supertrend[i-1])
                if close[i] < supertrend[i-1]:
                    supertrend[i] = lower_band
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band
                    direction[i] = 1
    
    return supertrend, direction


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    min_period = 80  # Wait for daily HMA and Supertrend to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(supertrend[i]) or 
            atr[i] == 0 or st_direction[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (price above/below daily HMA50)
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # RSI filter (avoid extremes - don't buy overbought or sell oversold)
        rsi_valid = 40 < rsi[i] < 60
        
        # Supertrend direction signal
        st_signal = int(st_direction[i])
        prev_st_signal = int(st_direction[i-1]) if i > 0 else 0
        
        # Detect Supertrend flip (entry signal)
        supertrend_flip = 0
        if prev_st_signal == -1 and st_signal == 1:
            supertrend_flip = 1  # Bullish flip
        elif prev_st_signal == 1 and st_signal == -1:
            supertrend_flip = -1  # Bearish flip
        
        # Determine target signal based on trend filter and RSI
        target_signal = 0.0
        if supertrend_flip == daily_trend and rsi_valid:
            target_signal = SIZE * supertrend_flip
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals