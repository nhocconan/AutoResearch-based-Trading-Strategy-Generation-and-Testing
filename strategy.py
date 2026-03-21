#!/usr/bin/env python3
"""
EXPERIMENT #003 - Supertrend + RSI Pullback + 4h HMA Trend (1h primary)
=====================================================================================
Hypothesis: 1h Supertrend(10,3) identifies the primary trend direction, while RSI(14)
pullbacks to 40-60 range provide optimal entry points within that trend. 4h HMA(21)
filter ensures we only trade with the higher timeframe trend. This combination
should generate more trades than pure breakout strategies while maintaining good
risk/reward ratios.

Key features:
- Primary TF: 1h (this experiment)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: Supertrend(10, 3.0) for direction
- Entry: RSI(14) pullback to 40-60 zone within trend
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work:
- 1h captures more opportunities than 4h but fewer false signals than 15m/30m
- Supertrend = clear trend direction with built-in stop
- RSI pullback = enter on dips within trend (better than chasing breakouts)
- 4h HMA filter = removes counter-trend trades
- Conservative sizing controls drawdown during crypto crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_4hhma_1h_v1"
timeframe = "1h"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period - 1, n):
        if atr[i] == 0:
            upper_band[i] = high[i]
            lower_band[i] = low[i]
        else:
            upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
            lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period - 1:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            if trend[i - 1] == 1:
                if close[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    supertrend[i] = max(lower_band[i], supertrend[i - 1])
                    trend[i] = 1
            else:
                if close[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    supertrend[i] = min(upper_band[i], supertrend[i - 1])
                    trend[i] = -1
    
    # Fill early bars
    supertrend[:period - 1] = np.nan
    trend[:period - 1] = 0
    
    return supertrend, trend


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    rsi[:period] = np.nan
    return rsi


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, period=14)
    supertrend, st_trend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # Supertrend direction
        st_direction = st_trend[i]  # 1 = bullish, -1 = bearish
        
        # RSI pullback zone (40-60 for entries within trend)
        rsi_in_zone = 40 <= rsi[i] <= 60
        rsi_oversold = rsi[i] < 40  # Strong long signal
        rsi_overbought = rsi[i] > 60  # Strong short signal
        
        # Calculate position size
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend bullish + 4h HMA bullish + RSI pullback or oversold
        if (st_direction == 1 and hma_trend == 1 and (rsi_in_zone or rsi_oversold)):
            target_signal = position_size
        
        # Short entry: Supertrend bearish + 4h HMA bearish + RSI pullback or overbought
        elif (st_direction == -1 and hma_trend == -1 and (rsi_in_zone or rsi_overbought)):
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
                
                # Check take profit (2R from entry, where R = 2.0*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4.0*ATR
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
                # Exit if Supertrend reverses OR 4h HMA alignment breaks
                st_reversal = (position_side == 1 and st_direction == -1) or \
                              (position_side == -1 and st_direction == 1)
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if st_reversal or hma_alignment_broken:
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