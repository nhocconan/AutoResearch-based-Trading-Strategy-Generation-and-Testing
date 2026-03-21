#!/usr/bin/env python3
"""
EXPERIMENT #004 - HMA Trend (12h) + Daily Regime + RSI Pullback Entry
======================================================================
Hypothesis: 12h timeframe provides cleaner trends than 4h/6h with fewer whipsaws.
Using Daily SMA(50) as regime filter ensures we only trade in direction of long-term trend.
RSI(14) pullback entries (RSI<45 for longs, RSI>55 for shorts) improve entry timing vs blind trend following.
ATR-based trailing stop protects capital during reversals.

Why this differs from failed strategies:
- 12h primary TF = even cleaner than 6h ( Experiment #004 Phase A focus)
- Daily regime filter = avoids counter-trend trades that caused drawdowns
- RSI pullback entry = better timing than pure breakout (reduces fee churn)
- Trailing stop logic = locks in profits, limits drawdown
- Conservative position size (0.35) with discrete levels

Key risk controls:
- Signal magnitude: 0.35 (35% position size)
- Stoploss: 2.5*ATR trailing stop from entry/highest
- Take profit: reduce to half at 2R, trail stop at 1R
- Discrete levels: 0.0, ±0.35, ±0.175 (half position)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_12h_daily_regime_rsi_v1"
timeframe = "12h"
leverage = 1.0


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, adjust=False, min_periods=half).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    wma_diff = 2 * wma1 - wma2
    
    hma = pd.Series(wma_diff).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    # Set first period values to NaN
    rsi[:period] = np.nan
    
    return rsi


def calculate_atr(high, low, close, period=14):
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (for regime filter)
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate daily SMA(50) for long-term trend regime
    daily_sma50 = pd.Series(daily_close).rolling(window=50, min_periods=50).mean().values
    daily_sma50_aligned = align_htf_to_ltf(prices, df_1d, daily_sma50)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate indicators on primary 12h timeframe
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Generate signals with discrete position sizing and stoploss
    signals = np.zeros(n)
    SIZE = 0.35  # 35% position size
    HALF_SIZE = SIZE / 2  # 17.5% for take profit reduction
    ATR_STOP_MULT = 2.5  # Stoploss at 2.5*ATR
    RSI_LONG_ENTRY = 45  # RSI pullback level for longs
    RSI_SHORT_ENTRY = 55  # RSI pullback level for shorts
    
    # Track position state
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0  # For trailing stop
    lowest_since_entry = 0.0   # For trailing stop
    tp_hit = False  # Track if take profit was hit
    
    # Find first valid index (all indicators ready)
    first_valid = max(50, 48, 14)  # Daily SMA(50), HMA(48), RSI(14)
    
    for i in range(first_valid, n):
        # Check for NaN values
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            continue
        
        # Daily regime filter
        daily_trend_bullish = False
        daily_trend_bearish = False
        
        if not np.isnan(daily_close_aligned[i]) and not np.isnan(daily_sma50_aligned[i]):
            daily_trend_bullish = daily_close_aligned[i] > daily_sma50_aligned[i]
            daily_trend_bearish = daily_close_aligned[i] < daily_sma50_aligned[i]
        
        # HMA trend direction on 12h
        hma_bullish = hma_21[i] > hma_48[i]
        hma_bearish = hma_21[i] < hma_48[i]
        
        # Check stoploss/trailing stop first (before new signals)
        if position_side == 1 and entry_price > 0:
            # Update highest since entry for trailing
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            # Trailing stop: 2.5*ATR from highest (for longs)
            trailing_stop = highest_since_entry - ATR_STOP_MULT * atr[i]
            
            # Initial stoploss: 2.5*ATR below entry
            initial_stop = entry_price - ATR_STOP_MULT * atr[i]
            stop_level = max(initial_stop, trailing_stop)
            
            # Check if stoploss hit
            if close[i] < stop_level:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                tp_hit = False
                continue
            
            # Take profit: reduce to half at 2R (2 * 2.5*ATR = 5*ATR from entry)
            tp_level = entry_price + 5.0 * atr[i]
            if not tp_hit and close[i] > tp_level:
                signals[i] = HALF_SIZE
                tp_hit = True
                # Move stop to breakeven + 1R
                highest_since_entry = max(highest_since_entry, close[i])
                continue
            
            # Maintain position
            signals[i] = HALF_SIZE if tp_hit else SIZE
            continue
        
        if position_side == -1 and entry_price > 0:
            # Update lowest since entry for trailing
            if close[i] < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = close[i]
            
            # Trailing stop: 2.5*ATR from lowest (for shorts)
            trailing_stop = lowest_since_entry + ATR_STOP_MULT * atr[i]
            
            # Initial stoploss: 2.5*ATR above entry
            initial_stop = entry_price + ATR_STOP_MULT * atr[i]
            stop_level = min(initial_stop, trailing_stop)
            
            # Check if stoploss hit
            if close[i] > stop_level:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                tp_hit = False
                continue
            
            # Take profit: reduce to half at 2R
            tp_level = entry_price - 5.0 * atr[i]
            if not tp_hit and close[i] < tp_level:
                signals[i] = -HALF_SIZE
                tp_hit = True
                lowest_since_entry = min(lowest_since_entry, close[i])
                continue
            
            # Maintain position
            signals[i] = -HALF_SIZE if tp_hit else -SIZE
            continue
        
        # Generate new entry signals (only if flat)
        if position_side == 0:
            # Long entry: HMA bullish + Daily bullish + RSI pullback
            if hma_bullish and (daily_trend_bullish or np.isnan(daily_sma50_aligned[i])):
                if rsi[i] < RSI_LONG_ENTRY:
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    tp_hit = False
                    continue
            
            # Short entry: HMA bearish + Daily bearish + RSI pullback
            if hma_bearish and (daily_trend_bearish or np.isnan(daily_sma50_aligned[i])):
                if rsi[i] > RSI_SHORT_ENTRY:
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
                    tp_hit = False
                    continue
        
        # No signal - maintain or flatten
        if position_side == 0:
            signals[i] = 0.0
        else:
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals