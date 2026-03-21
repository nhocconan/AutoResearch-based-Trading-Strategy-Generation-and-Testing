#!/usr/bin/env python3
"""
EXPERIMENT #020 - KAMA Adaptive Trend + RSI Pullback with 4h Filter (30m)
==========================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility,
providing better trend signals in choppy conditions than static EMAs. Combined
with 4h KAMA for major trend direction and 30m RSI pullback entries, this
captures trend continuations at favorable prices. Volume confirmation filters
false breakouts. ATR-based stoploss and position sizing control drawdown.

Key features:
- Primary TF: 30m (required for this experiment)
- HTF filter: 4h KAMA(21) for major trend direction
- Entry: RSI(14) pullback to 40-60 zone in trend direction
- Filter: 30m price must be above/below 30m KAMA(21) for confirmation
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.125)
- Take profit: Reduce to half at 2R profit

Why this differs from failures:
- #014 supertrend_rsi_4h_filter_30m_v1 had 0 trades (too restrictive filters)
- #013 mtf_rsi_pullback_4h_1h_15m_v1 had -54% DD (position sizing too aggressive)
- This uses KAMA (adaptive) instead of Supertrend/HMA, simpler RSI entry logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_4h_30m_v1"
timeframe = "30m"
leverage = 1.0


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period - 1, n):
        change = abs(close[i] - close[i - period + 1])
        volatility = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    kama_4h = calculate_kama(df_4h['close'].values, period=21)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 30m indicators
    kama_30m = calculate_kama(close, period=21)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    volume_sma = calculate_volume_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_30m[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(volume_sma[i]) or 
            atr[i] == 0 or kama_30m[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (HTF)
        htf_trend = 1 if close[i] > kama_4h_aligned[i] else -1
        
        # 30m trend confirmation
        ltf_trend = 1 if close[i] > kama_30m[i] else -1
        
        # Volume confirmation (must be above 20-period average)
        volume_confirmed = volume[i] > volume_sma[i]
        
        # RSI pullback entry logic
        # Long: 4h trend up, 30m trend up, RSI pulled back to 40-55 zone
        # Short: 4h trend down, 30m trend down, RSI rallied to 45-60 zone
        long_entry = (htf_trend == 1 and ltf_trend == 1 and 
                      40 <= rsi[i] <= 55 and volume_confirmed)
        short_entry = (htf_trend == -1 and ltf_trend == -1 and 
                       45 <= rsi[i] <= 60 and volume_confirmed)
        
        # Determine target signal based on entry conditions
        target_signal = 0.0
        if long_entry:
            target_signal = SIZE
        elif short_entry:
            target_signal = -SIZE
        
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
                    r_distance = 2.5 * entry_atr
                    if close[i] >= entry_price + 2.0 * r_distance:
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
                    r_distance = 2.5 * entry_atr
                    if close[i] <= entry_price - 2.0 * r_distance:
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
            if target_signal != 0.0:
                # Only enter if flat or reversing (avoid churning)
                if position_side == 0 or np.sign(target_signal) != np.sign(position_side):
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    entry_atr = atr[i]
                    profit_target_hit = False
                else:
                    # Maintain existing position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals