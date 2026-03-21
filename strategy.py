#!/usr/bin/env python3
"""
EXPERIMENT #015 - EMA Crossover + 12h Trend Filter + ATR Risk Management (1h primary)
=====================================================================================
Hypothesis: Previous strategies failed due to TOO MANY conflicting filters causing
either 0 trades or massive drawdowns. This strategy uses SIMPLER logic:
- 12h EMA(50) for major trend direction (smoother than 4h, fewer whipsaws)
- 1h EMA(8)/EMA(21) crossover for entry timing (generates regular signals)
- RSI(14) moderate filter only (avoid extreme overbought/oversold)
- 2.5*ATR trailing stoploss (critical for drawdown control)
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25)

Why this should work better than failed strategies:
- Fewer filters = more trades (avoiding #012's 8-trade failure)
- 12h trend is smoother than 4h (avoiding #003's -51% DD)
- No Supertrend/BB/KAMA complexity (avoiding #007-#009's 80-99% DD)
- Simple EMA crossover generates 100-500 trades range (sweet spot)
- Conservative sizing (0.25) controls drawdown during 2022 crash

Key improvements:
- Removed volume filter (too restrictive, caused missed entries)
- Relaxed RSI to 25-75 range (allows trend continuation entries)
- Using 12h instead of 4h/1d for balanced trend filter
- Strict 2.5*ATR stoploss on every position
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_cross_12htrend_atr_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(50) for major trend filter
    ema_12h = calculate_ema(df_12h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1h indicators
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% of capital per position (conservative)
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(ema_8[i]) or
            np.isnan(ema_21[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 12h EMA trend filter
        price_above_12h = close[i] > ema_12h_aligned[i]
        trend_bullish = price_above_12h
        trend_bearish = not price_above_12h
        
        # EMA crossover signals (current vs previous bar)
        ema_bullish_cross = ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1]
        ema_bearish_cross = ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1]
        
        # RSI filter (moderate - avoid extreme overbought/oversold only)
        rsi_not_extreme = 25 < rsi[i] < 75
        
        # Stoploss and take profit check - evaluate BEFORE new entries
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest since entry
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
            else:
                # Short position - update lowest since entry
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        # Execute stoploss/takeprofit first
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit, trail stop continues
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Entry logic - only when flat
            target_signal = 0.0
            
            # Long: bullish 12h trend + EMA cross + RSI not extreme
            if trend_bullish and ema_bullish_cross and rsi_not_extreme:
                target_signal = BASE_SIZE
            
            # Short: bearish 12h trend + EMA cross + RSI not extreme
            elif trend_bearish and ema_bearish_cross and rsi_not_extreme:
                target_signal = -BASE_SIZE
            
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
                # Maintain existing position - check for exit conditions
                ema_reversal_long = ema_8[i] < ema_21[i]
                ema_reversal_short = ema_8[i] > ema_21[i]
                trend_broken = (position_side == 1 and trend_bearish) or \
                               (position_side == -1 and trend_bullish)
                
                if ema_reversal_long or ema_reversal_short or trend_broken:
                    # Exit on EMA reversal or trend break
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals