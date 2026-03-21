#!/usr/bin/env python3
"""
EXPERIMENT #004 - KAMA Adaptive Trend with Daily Filter (4h)
============================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility
better than static EMAs, reducing whipsaws in choppy markets. Combined with
1d HMA trend filter and Bollinger Band width regime detection, we only trade
when volatility expands (breakouts) in direction of higher timeframe trend.

Key features:
- Primary TF: 4h (4-hour candles)
- HTF filter: 1d HMA(50) for major trend direction
- Entry: KAMA(10) crossover + Bollinger expansion
- Filter: BB Width > 20th percentile (avoid squeeze/chop)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_bb_daily_filter_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        sum_volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if sum_volatility > 0:
            er[i] = price_change / sum_volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma
    return upper.values, lower.values, width.values


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


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
    kama = calculate_kama(close, period=10)
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
    # Calculate BB width percentile for regime filter
    bb_width_pct = np.full(n, np.nan)
    lookback = 100
    for i in range(lookback, n):
        valid_widths = bb_width[i-lookback:i+1]
        valid_widths = valid_widths[~np.isnan(valid_widths)]
        if len(valid_widths) > 0:
            bb_width_pct[i] = np.sum(valid_widths < bb_width[i]) / len(valid_widths)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    min_period = 80  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(atr[i]) or np.isnan(bb_width[i]) or 
            np.isnan(rsi[i]) or np.isnan(bb_width_pct[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF)
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # Bollinger regime filter (only trade when volatility expands)
        # BB width percentile > 0.3 means we're not in tight squeeze
        regime_valid = bb_width_pct[i] > 0.30
        
        # KAMA crossover signal
        kama_signal = 0
        if i >= 2:
            if not np.isnan(kama[i-1]) and not np.isnan(kama[i-2]):
                # Price crosses above KAMA
                if close[i-1] <= kama[i-1] and close[i] > kama[i]:
                    kama_signal = 1
                # Price crosses below KAMA
                elif close[i-1] >= kama[i-1] and close[i] < kama[i]:
                    kama_signal = -1
        
        # RSI filter (avoid entering at extremes)
        rsi_valid = 30 < rsi[i] < 70
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if kama_signal != 0 and regime_valid and rsi_valid:
            # Only trade in direction of daily trend
            if kama_signal == daily_trend:
                target_signal = SIZE * kama_signal
        
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
            entry_price = 0.0
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                elif position_side != (1 if target_signal > 0 else -1):
                    # Reversal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals