#!/usr/bin/env python3
"""
EXPERIMENT #004 - Supertrend + Daily Trend Filter + Volume Confirmation (4h)
=============================================================================
Hypothesis: 4h Supertrend breakouts aligned with 1d HMA(50) trend direction
capture sustained moves while filtering counter-trend noise. Volume confirmation
reduces false breakouts. ATR trailing stop protects capital during reversals.

Key features:
- Primary TF: 4h (4-hour candles)
- HTF filter: 1d HMA(50) for major trend direction
- Entry: Supertrend(10,3) flip + volume > 20-period average
- Filter: Daily trend must align with Supertrend direction
- RSI(14) filter: avoid extreme overbought/oversold entries
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this differs from failed strategies:
- #001 used 15m primary (too noisy) - we use 4h (cleaner signals)
- #002 had 0 trades (too restrictive) - we relax filters slightly
- #003 had negative Sharpe - we add daily trend filter for direction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_daily_volume_4h_v1"
timeframe = "4h"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    # Calculate basic upper/lower bands
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    direction[0] = 1  # Start assuming uptrend
    
    for i in range(1, n):
        if direction[i - 1] == 1:
            # Previous trend was up
            if close[i] > supertrend[i - 1]:
                # Trend continues up
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
                direction[i] = 1
            else:
                # Trend flips down
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previous trend was down
            if close[i] < supertrend[i - 1]:
                # Trend continues down
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
                direction[i] = -1
            else:
                # Trend flips up
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction


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
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    supertrend_vals, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Volume moving average
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for daily HMA and indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(supertrend_dir[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma[i]) or 
            np.isnan(rsi[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (1d HMA50)
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # Volume confirmation (must be above 20-period average)
        volume_confirmed = volume[i] > volume_sma[i]
        
        # RSI filter (avoid extreme overbought/oversold entries)
        rsi_valid = 25 < rsi[i] < 75
        
        # Supertrend direction signal
        supertrend_signal = int(supertrend_dir[i])
        
        # Detect Supertrend flip (entry signal)
        supertrend_flip = 0
        if i > 0:
            if supertrend_dir[i] != supertrend_dir[i - 1]:
                supertrend_flip = int(supertrend_dir[i])
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if supertrend_flip != 0:
            # Supertrend flip must align with daily trend
            if supertrend_flip == daily_trend and volume_confirmed and rsi_valid:
                target_signal = SIZE * supertrend_flip
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
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
                    profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals