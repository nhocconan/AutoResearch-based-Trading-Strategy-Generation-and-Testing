#!/usr/bin/env python3
"""
EXPERIMENT #001 - HMA Crossover + Z-Score Filter + 1h Trend (15m primary)
================================================================================
Hypothesis: 15m HMA(8)/HMA(21) crossover provides timely entries, but needs
HTF trend filter to avoid counter-trend whipsaws. Z-score(20) filter prevents
entering at extreme overbought/oversold levels. 1h HMA(50) provides major
trend alignment. This differs from supertrend by using smoother HMA signals
and mean-reversion filter for better entry timing.

Key features:
- Primary TF: 15m (faster entries than 4h strategies)
- HTF filter: 1h HMA(50) for trend direction
- Trend: HMA(8) vs HMA(21) crossover on 15m
- Entry filter: Z-score(20) between -1.5 and +1.5 (avoid extremes)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_crossover_zscore_15m_1h_v1"
timeframe = "15m"
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1h HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    hma_1h = calculate_hma(df_1h['close'].values, 50)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30  # Entry position size (30% of capital)
    SIZE_HALF = 0.15   # Half position for take profit
    SIZE_EXIT = 0.0    # Flat
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 1.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(hma_fast[i]) or 
            np.isnan(hma_slow[i]) or np.isnan(atr[i]) or np.isnan(zscore[i]) or 
            atr[i] == 0):
            signals[i] = SIZE_EXIT
            continue
        
        # 1h trend filter (HTF)
        hourly_trend = 1 if close[i] > hma_1h_aligned[i] else -1
        
        # 15m HMA crossover signal
        hma_cross = 0
        if hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]:
            hma_cross = 1  # Bullish crossover
        elif hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]:
            hma_cross = -1  # Bearish crossover
        elif hma_fast[i] > hma_slow[i]:
            hma_cross = 1  # Already bullish
        elif hma_fast[i] < hma_slow[i]:
            hma_cross = -1  # Already bearish
        
        # Z-score filter: avoid entering at extremes (|z| > 1.5)
        zscore_valid = abs(zscore[i]) <= 1.5
        
        # Determine target signal based on all filters
        target_signal = SIZE_EXIT
        
        # Long entry: HMA bullish + 1h trend bullish + Z-score valid
        if hma_cross == 1 and hourly_trend == 1 and zscore_valid:
            target_signal = SIZE_ENTRY
        
        # Short entry: HMA bearish + 1h trend bearish + Z-score valid
        elif hma_cross == -1 and hourly_trend == -1 and zscore_valid:
            target_signal = -SIZE_ENTRY
        
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
                
                # Check take profit (2R from entry, where R = 2.0*ATR)
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
            signals[i] = SIZE_EXIT
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 1.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = SIZE_HALF * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != SIZE_EXIT and position_side == 0:
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
                if position_side == 1 and hma_cross == -1:
                    # HMA crossed bearish, exit long
                    signals[i] = SIZE_EXIT
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 1.0
                    profit_target_hit = False
                elif position_side == -1 and hma_cross == 1:
                    # HMA crossed bullish, exit short
                    signals[i] = SIZE_EXIT
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 1.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE_ENTRY * position_side if not profit_target_hit else SIZE_HALF * position_side
            else:
                signals[i] = SIZE_EXIT
    
    return signals