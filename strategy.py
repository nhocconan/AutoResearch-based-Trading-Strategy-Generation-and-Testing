#!/usr/bin/env python3
"""
EXPERIMENT #016 - HMA Crossover + Z-Score Entries + 1d Trend Filter (4h primary)
=====================================================================================
Hypothesis: 4h HMA crossover captures medium-term trends, but entry timing is poor.
Adding Z-score(20) for mean reversion entries within the trend improves risk/reward.
1d HMA(21) filter ensures we only trade with the weekly/major trend direction.

Key features:
- Primary TF: 4h (captures swing moves, less noise than 1h/15m)
- HTF filter: 1d HMA(21) for major trend direction
- Trend: HMA(21) vs HMA(55) crossover on 4h
- Entry: Z-score(20) extremes within trend (Z < -1.0 long, Z > 1.0 short)
- Exit: HMA crossover reversal OR 2*ATR stoploss
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work:
- 4h timeframe balances signal frequency vs noise (unlike 15m which was too noisy)
- HMA is faster than EMA, captures trends earlier
- Z-score entries avoid chasing breakouts (buy dips in uptrend)
- 1d filter removes counter-trend trades that caused massive DD in prior experiments
- Conservative sizing (0.25-0.30) controls drawdown during crypto crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_zscore_1dtrend_4h_v1"
timeframe = "4h"
leverage = 1.0


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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    half_period = period // 2
    if half_period < 1:
        half_period = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    wma1 = close_s.ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    zscore = zscore.fillna(0).values
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend filter
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    hma_fast = calculate_hma(close, 21)
    hma_slow = calculate_hma(close, 55)
    atr = calculate_atr(high, low, close, period=14)
    zscore = calculate_zscore(close, period=20)
    
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
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_fast[i]) or
            np.isnan(hma_slow[i]) or np.isnan(atr[i]) or np.isnan(zscore[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1d HMA trend filter (major trend direction)
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        hma_1d_trend = 1 if price_above_1d_hma else -1
        
        # 4h HMA crossover trend
        hma_crossover_bullish = hma_fast[i] > hma_slow[i]
        hma_crossover_bearish = hma_fast[i] < hma_slow[i]
        
        # Z-score entry signals (mean reversion within trend)
        # Long: Z-score < -1.0 (price below mean) in uptrend
        # Short: Z-score > 1.0 (price above mean) in downtrend
        zscore_oversold = zscore[i] < -1.0
        zscore_overbought = zscore[i] > 1.0
        
        # Calculate position size (dynamic based on Z-score magnitude)
        zscore_magnitude = abs(zscore[i])
        if zscore_magnitude > 2.0:
            position_size = MAX_SIZE
        elif zscore_magnitude > 1.5:
            position_size = BASE_SIZE
        else:
            position_size = MIN_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h HMA bullish + 1d HMA bullish + Z-score oversold
        if hma_crossover_bullish and hma_1d_trend == 1 and zscore_oversold:
            target_signal = position_size
        
        # Short entry: 4h HMA bearish + 1d HMA bearish + Z-score overbought
        elif hma_crossover_bearish and hma_1d_trend == -1 and zscore_overbought:
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
                # Exit if HMA crossover reverses OR 1d HMA alignment breaks
                hma_reversal_long = hma_crossover_bearish
                hma_reversal_short = hma_crossover_bullish
                hma_1d_alignment_broken = (position_side == 1 and hma_1d_trend == -1) or \
                                          (position_side == -1 and hma_1d_trend == 1)
                
                if hma_reversal_long or hma_reversal_short or hma_1d_alignment_broken:
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