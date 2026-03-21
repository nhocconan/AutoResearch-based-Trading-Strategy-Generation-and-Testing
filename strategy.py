#!/usr/bin/env python3
"""
EXPERIMENT #011 - KAMA Adaptive Trend with Weekly Filter (12h)
==============================================================
Hypothesis: KAMA adapts to market noise better than EMA/HMA, combining with
Bollinger Band Width regime detection to avoid trading during low-volatility
chop. Weekly HMA(50) ensures we trade with the major trend. RSI filters entries
to avoid chasing extremes. ATR trailing stop protects capital.

Key features:
- Primary TF: 12h (12-hour candles)
- HTF filter: 1w HMA(50) for major trend direction
- Trend indicator: KAMA(ER=10, fast=2, slow=30) - adapts to volatility
- Regime filter: Bollinger Band Width > 20th percentile (avoid squeeze)
- Entry: KAMA crossover + RSI(14) between 40-60
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_bb_weekly_regime_12h_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average
    KAMA adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


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


def calculate_atr(high, low, close, period=14):
    """Calculate ATR"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


def calculate_percentile_rank(series, lookback=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(lookback, n):
        window = series[i - lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < series[i]) / len(valid)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=20)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    min_period = 120  # Wait for weekly HMA and indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(kama_fast[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(bb_width_pr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # Bollinger Band Width regime filter (avoid low volatility squeeze)
        # Only trade when BB Width is above 20th percentile of last 100 bars
        regime_valid = bb_width_pr[i] > 0.20
        
        # KAMA crossover signal
        kama_signal = 0
        if i > 0:
            if kama_fast[i] > kama[i] and kama_fast[i - 1] <= kama[i - 1]:
                kama_signal = 1  # Bullish crossover
            elif kama_fast[i] < kama[i] and kama_fast[i - 1] >= kama[i - 1]:
                kama_signal = -1  # Bearish crossover
        
        # RSI filter (avoid extremes - enter in neutral zone)
        rsi_valid = 40 < rsi[i] < 60
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if kama_signal == weekly_trend and regime_valid and rsi_valid:
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
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals