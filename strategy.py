#!/usr/bin/env python3
"""
EXPERIMENT #008 - MACD Momentum with 4h Trend + Bollinger Regime (30m)
=================================================================
Hypothesis: 30m MACD histogram momentum entries filtered by 4h HMA trend
and Bollinger Band width regime detection. Only trade when:
1. 4h HMA confirms trend direction
2. Bollinger Band width > median (avoid squeeze/chop)
3. MACD histogram shows momentum shift
4. Volume confirms the move

This combines momentum + trend + regime filtering for higher quality signals.
Position sizing: 0.20-0.30 discrete, stoploss at 2*ATR trailing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_momentum_4h_regime_30m_v1"
timeframe = "30m"
leverage = 1.0


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate median BB width for regime filter
    bb_width_valid = bb_width[~np.isnan(bb_width)]
    bb_width_median = np.median(bb_width_valid) if len(bb_width_valid) > 0 else 0.05
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30  # Entry position size (30% of capital)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    min_period = 60  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(histogram[i]) or 
            np.isnan(atr[i]) or np.isnan(bb_width[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Bollinger regime filter (avoid squeeze)
        regime_valid = bb_width[i] > bb_width_median * 0.7
        
        # MACD momentum signal (histogram crossover)
        momentum_signal = 0
        if histogram[i] > 0 and histogram[i-1] <= 0:
            momentum_signal = 1  # Bullish momentum shift
        elif histogram[i] < 0 and histogram[i-1] >= 0:
            momentum_signal = -1  # Bearish momentum shift
        
        # Volume confirmation (volume > 80% of recent average)
        vol_avg = np.mean(volume[max(0, i-20):i+1])
        volume_confirmed = volume[i] > vol_avg * 0.8
        
        # Determine target signal
        target_signal = 0.0
        if momentum_signal == trend_4h and regime_valid and volume_confirmed:
            target_signal = SIZE_ENTRY * momentum_signal
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE_ENTRY * position_side
            else:
                signals[i] = 0.0
    
    return signals