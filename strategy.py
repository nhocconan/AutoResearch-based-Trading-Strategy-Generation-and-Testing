#!/usr/bin/env python3
"""
EXPERIMENT #069 - MACD Momentum + Bollinger Regime + 4h Trend Filter (1h primary)
=====================================================================================
Hypothesis: MACD histogram momentum captures trend acceleration, but fails in chop.
Bollinger Band width percentile detects regime (squeeze=low vol, expansion=high vol).
4h HMA trend filter ensures we trade with higher timeframe direction. This differs from
failed strategies by using MACD momentum (not HMA/RSI pullback) + BB regime (not ADX).

Key features:
- Primary TF: 1h
- HTF filter: 4h HMA(21) for trend direction
- Momentum: MACD(12,26,9) histogram turning points
- Regime: BB width percentile > 40th (avoid extreme squeeze)
- Entry: MACD hist crosses zero + BB regime + 4h trend alignment
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should beat current best (Sharpe=0.490):
- MACD momentum works well on 1h timeframe for crypto
- BB regime filter avoids low-volatility traps
- 4h trend filter removes counter-trend trades
- Conservative sizing (0.25-0.30) controls drawdown
- Different from 66 failed strategies (no HMA/RSI pullback pattern)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_bbregime_4htrend_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
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
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values, std.values


def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width (normalized)"""
    width = (upper - lower) / sma
    return width


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_std = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    
    # Calculate BB width percentile rank (regime filter)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(macd_line[i]) or
            np.isnan(signal_line[i]) or np.isnan(histogram[i]) or
            np.isnan(atr[i]) or np.isnan(bb_width_pr[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # BB regime filter (avoid extreme squeeze < 40th percentile)
        bb_regime_ok = bb_width_pr[i] > 0.40
        
        # MACD momentum signals
        macd_bullish = histogram[i] > 0 and histogram[i-1] <= 0  # Cross above zero
        macd_bearish = histogram[i] < 0 and histogram[i-1] >= 0  # Cross below zero
        
        # MACD histogram acceleration (momentum strengthening)
        macd_hist_increasing = histogram[i] > histogram[i-1]
        macd_hist_decreasing = histogram[i] < histogram[i-1]
        
        # Price position within BB
        price_in_upper_bb = close[i] > bb_sma[i]
        price_in_lower_bb = close[i] < bb_sma[i]
        
        # Calculate position size (discrete levels)
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: MACD bullish cross + BB regime ok + 4h trend up + price above SMA
        if (macd_bullish and bb_regime_ok and hma_trend == 1 and price_in_upper_bb):
            target_signal = position_size
        
        # Short entry: MACD bearish cross + BB regime ok + 4h trend down + price below SMA
        elif (macd_bearish and bb_regime_ok and hma_trend == -1 and price_in_lower_bb):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if MACD reverses OR 4h trend breaks
                macd_reversal_long = histogram[i] < 0
                macd_reversal_short = histogram[i] > 0
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if macd_reversal_long or macd_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals