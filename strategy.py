#!/usr/bin/env python3
"""
EXPERIMENT #006 - HMA Crossover + RSI Momentum + 1w Trend Filter (1d primary)
=====================================================================================
Hypothesis: Daily timeframe captures major crypto trends while avoiding noise of lower TFs.
HMA crossover (8/21) provides fast trend detection with minimal lag. RSI(14) momentum
confirms entry direction without requiring extreme values (which cause 0 trades).
Weekly HMA(21) filter ensures we trade with the major trend, reducing whipsaws.

Key features:
- Primary TF: 1d (daily candles - fewer but higher quality signals)
- HTF filter: 1w HMA(21) for major trend direction
- Trend: HMA(8) vs HMA(21) crossover
- Entry: RSI(14) momentum confirmation (RSI > 50 long, RSI < 50 short)
- Stoploss: 2.5*ATR(14) trailing (wider for daily TF)
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work on 1d:
- Daily candles filter out intraday noise that caused false signals on 15m/1h
- HMA crossover generates ~20-50 signals/year on crypto (enough for 10+ trades)
- Weekly trend filter removes counter-trend trades during major moves
- Conservative sizing (0.25-0.30) protects against 70%+ crypto crashes
- RSI > 50/< 50 is less strict than RSI < 45/> 55 (avoids 0 trade problem)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_crossover_rsi_1wtrend_1d_v1"
timeframe = "1d"
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
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


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
    
    # Use EMA for smoothing (Wilder's method)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend filter
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong momentum
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for indicators to stabilize (less than 15m strategies)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(hma_fast[i]) or
            np.isnan(hma_slow[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1w HMA trend filter (major trend direction)
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # HMA crossover signal
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # Check for crossover (fast crossed above/below slow)
        hma_cross_long = False
        hma_cross_short = False
        
        if i > 0:
            prev_hma_bullish = hma_fast[i-1] > hma_slow[i-1]
            prev_hma_bearish = hma_fast[i-1] < hma_slow[i-1]
            
            # Bullish crossover: was bearish, now bullish
            if prev_hma_bearish and hma_bullish:
                hma_cross_long = True
            # Bearish crossover: was bullish, now bearish
            if prev_hma_bullish and hma_bearish:
                hma_cross_short = True
        
        # RSI momentum confirmation (less strict than pullback strategy)
        rsi_bullish = rsi[i] > 50  # Momentum to the upside
        rsi_bearish = rsi[i] < 50  # Momentum to the downside
        
        # Calculate position size based on RSI strength
        rsi_strength = abs(rsi[i] - 50) / 50  # 0 to 1
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * (1.0 + rsi_strength * 0.25)))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HMA bullish crossover + 1w HMA bullish + RSI > 50
        if hma_cross_long and weekly_trend == 1 and rsi_bullish:
            target_signal = position_size
        
        # Short entry: HMA bearish crossover + 1w HMA bearish + RSI < 50
        elif hma_cross_short and weekly_trend == -1 and rsi_bearish:
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]  # Wider stop for daily
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
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
                # Exit if HMA crossover reverses OR 1w HMA alignment breaks
                hma_reversal_long = hma_bearish  # Fast below slow
                hma_reversal_short = hma_bullish  # Fast above slow
                weekly_alignment_broken = (position_side == 1 and weekly_trend == -1) or \
                                          (position_side == -1 and weekly_trend == 1)
                
                if hma_reversal_long or hma_reversal_short or weekly_alignment_broken:
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