#!/usr/bin/env python3
"""
EXPERIMENT #026 - HMA Crossover + RSI Momentum + 4h Trend Filter (30m primary)
================================================================================
Hypothesis: Hull Moving Average crossovers provide faster trend detection than 
traditional EMA crossovers with less lag. RSI momentum confirmation ensures we 
enter when momentum supports the trend direction. 4h HMA(50) provides major 
trend alignment to avoid counter-trend trades. ADX filters out weak/choppy markets.

Key features:
- Primary TF: 30m (REQUIRED for this experiment)
- HTF filter: 4h HMA(50) for major trend direction
- Trend: HMA(8) / HMA(21) crossover on 30m
- Momentum: RSI(14) > 55 for long, < 45 for short
- Regime: ADX(14) > 20 for trend strength
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_crossover_rsi_30m_4h_v1"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average - reduces lag vs EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) - trend strength"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1) for completed bars
    
    # Calculate 30m indicators (pre-compute before loop for performance)
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% of capital - conservative sizing
    HALF_SIZE = SIZE / 2  # 14% for take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    profit_target_hit = False
    
    min_period = 120  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_fast[i]) or 
            np.isnan(hma_slow[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (HTF) - only trade with major trend
        daily_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # ADX trend strength filter - avoid choppy markets
        trend_strength = adx[i] > 20
        
        # HMA crossover signals
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # RSI momentum confirmation
        rsi_long = rsi[i] > 55
        rsi_short = rsi[i] < 45
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HMA cross + RSI momentum + 4h trend + ADX strength
        if hma_cross_long and rsi_long and daily_trend == 1 and trend_strength:
            target_signal = SIZE
        
        # Short entry: HMA cross + RSI momentum + 4h trend + ADX strength
        elif hma_cross_short and rsi_short and daily_trend == -1 and trend_strength:
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
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
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
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position - check if trend reversed (opposite crossover)
                if position_side == 1 and hma_cross_short:
                    # Opposite crossover, exit long
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    profit_target_hit = False
                elif position_side == -1 and hma_cross_long:
                    # Opposite crossover, exit short
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    profit_target_hit = False
                else:
                    # Maintain position - same signal value = no fee churn
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals