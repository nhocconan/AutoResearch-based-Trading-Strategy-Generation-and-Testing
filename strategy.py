#!/usr/bin/env python3
"""
EXPERIMENT #054 - HMA Trend + RSI Pullback + Weekly Filter (1d primary)
========================================================================
Hypothesis: Daily HMA trend captures major moves, but entering on RSI pullbacks
(40-60 zone) gives better risk/reward than breakout entries. Weekly HMA filter
ensures we trade with the major trend. This differs from Donchian breakouts by
fading short-term counter-trend moves within the larger trend.

Key features:
- Primary TF: 1d
- HTF filter: 1w HMA(50) for major trend alignment
- Trend: HMA(21) vs HMA(50) crossover on daily
- Entry: RSI(14) pullback to 40-60 zone in trend direction
- Regime: Weekly HMA confirms major trend direction
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels

Why this should beat current best (Sharpe=0.490):
- Pullback entries have better R:R than breakouts
- Fewer signals = less fee churn on daily TF
- Weekly filter removes counter-trend trades in major reversals
- HMA is more responsive than SMA for trend detection
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_pullback_weekly_1d_1w_v1"
timeframe = "1d"
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


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period - 1, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_hma_crossover(hma_fast, hma_slow):
    """Detect HMA crossovers (1=fast crosses above slow, -1=fast crosses below)"""
    n = len(hma_fast)
    crossover = np.zeros(n)
    
    for i in range(1, n):
        if not np.isnan(hma_fast[i]) and not np.isnan(hma_slow[i]):
            if hma_fast[i - 1] <= hma_slow[i - 1] and hma_fast[i] > hma_slow[i]:
                crossover[i] = 1
            elif hma_fast[i - 1] >= hma_slow[i - 1] and hma_fast[i] < hma_slow[i]:
                crossover[i] = -1
    
    return crossover


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # HMA crossover signals
    hma_cross = calculate_hma_crossover(hma_21, hma_50)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
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
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or 
            np.isnan(hma_50[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (major trend direction)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend (HMA 21 vs 50)
        daily_bullish = hma_21[i] > hma_50[i]
        daily_bearish = hma_21[i] < hma_50[i]
        
        # RSI pullback zones
        rsi_pullback_long = 40 <= rsi[i] <= 60  # Pullback in uptrend
        rsi_pullback_short = 40 <= rsi[i] <= 60  # Pullback in downtrend
        rsi_extreme_long = rsi[i] < 35  # Oversold bounce
        rsi_extreme_short = rsi[i] > 65  # Overbought fade
        
        # Calculate position size
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Weekly bullish + Daily bullish + RSI pullback or extreme
        if (weekly_bullish and daily_bullish and 
            (rsi_pullback_long or rsi_extreme_long)):
            target_signal = position_size
        
        # Short entry: Weekly bearish + Daily bearish + RSI pullback or extreme
        elif (weekly_bearish and daily_bearish and 
              (rsi_pullback_short or rsi_extreme_short)):
            target_signal = -position_size
        
        # Exit signal: HMA crossover against position
        exit_long = hma_cross[i] == -1  # Fast crosses below slow
        exit_short = hma_cross[i] == 1  # Fast crosses above slow
        
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
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
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
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
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
            # Reduce position to half at 2.5R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        elif exit_long and position_side == 1:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif exit_short and position_side == -1:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
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
                # Check if trend reversed (weekly filter broken)
                trend_reversal = ((position_side == 1 and not weekly_bullish) or
                                  (position_side == -1 and not weekly_bearish))
                
                if trend_reversal:
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