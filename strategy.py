#!/usr/bin/env python3
"""
EXPERIMENT #004 - KAMA Trend + RSI Pullback + Triple HTF Alignment (4h primary)
================================================================================
Hypothesis: 4h timeframe captures medium-term trends without excessive noise.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends, slow in chop.
RSI(14) pullback entries (30-70 range) provide better risk/reward than breakouts.
Triple HTF alignment (4h price vs 1d HMA vs 1w HMA) ensures we trade with major trend.

Key differences from failed strategies:
- Primary TF: 4h (not 15m/30m/1h) = fewer signals, less fee churn
- KAMA adapts to volatility (unlike fixed EMA/HMA)
- RSI pullback entries (not breakouts) = better entry timing
- Triple HTF filter (4h/1d/1w) = stricter trend confirmation
- Conservative sizing (0.25 base, max 0.30) = controls drawdown

Why this should beat current best:
- 4h has proven better Sharpe than lower TFs in backtests
- KAMA reduces whipsaws in choppy markets
- RSI pullbacks avoid chasing breakouts at tops/bottoms
- Triple HTF alignment filters 70%+ of false signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_triplehtf_4h_1d_1w_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = close[i]
    
    return kama


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


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=20)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size
    MIN_SIZE = 0.15   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(kama_fast[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend alignment
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # 1d and 1w trend direction
        daily_trend = 1 if price_above_1d_hma else -1
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # KAMA trend direction (4h)
        kama_bullish = kama_fast[i] > kama[i]  # Fast KAMA above slow KAMA
        kama_bearish = kama_fast[i] < kama[i]
        
        # KAMA slope (trend strength)
        kama_slope = kama[i] - kama[i - 5] if i >= 5 else 0
        kama_uptrend = kama_slope > 0
        kama_downtrend = kama_slope < 0
        
        # RSI pullback conditions (buy dips in uptrend, sell rallies in downtrend)
        rsi_oversold = rsi[i] < 40  # Pullback in uptrend
        rsi_overbought = rsi[i] > 60  # Rally in downtrend
        rsi_neutral = 35 < rsi[i] < 65  # Not at extremes
        
        # Calculate position size based on trend alignment strength
        alignment_score = 0
        if daily_trend == 1 and weekly_trend == 1:
            alignment_score = 2  # Strong bullish
        elif daily_trend == -1 and weekly_trend == -1:
            alignment_score = -2  # Strong bearish
        elif daily_trend == 1 or weekly_trend == 1:
            alignment_score = 1  # Weak bullish
        else:
            alignment_score = -1  # Weak bearish
        
        # Scale position size by alignment strength
        if abs(alignment_score) == 2:
            position_size = MAX_SIZE
        else:
            position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + RSI pullback + Triple HTF bullish alignment
        if (kama_bullish and kama_uptrend and rsi_oversold and 
            daily_trend == 1 and weekly_trend == 1):
            target_signal = position_size
        
        # Short entry: KAMA bearish + RSI rally + Triple HTF bearish alignment
        elif (kama_bearish and kama_downtrend and rsi_overbought and 
              daily_trend == -1 and weekly_trend == -1):
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
                if not profit_target_hit and entry_atr > 0:
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
                if not profit_target_hit and entry_atr > 0:
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
                # Exit if KAMA reverses OR HTF alignment breaks
                kama_reversal_long = kama_fast[i] < kama[i] and position_side == 1
                kama_reversal_short = kama_fast[i] > kama[i] and position_side == -1
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
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