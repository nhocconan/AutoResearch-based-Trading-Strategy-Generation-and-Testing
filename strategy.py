#!/usr/bin/env python3
"""
EXPERIMENT #003 - EMA Crossover + RSI Pullback + 4h HMA Trend Filter (1h primary)
=====================================================================================
Hypothesis: 1h EMA crossovers generate many signals but most fail in chop. 
RSI pullback entries (waiting for RSI to dip in uptrend / rise in downtrend) 
filter out 50%+ of false breakouts. 4h HMA(50) ensures we trade with the 
major trend direction. Volume confirmation adds extra filter for conviction.

Key features:
- Primary TF: 1h
- HTF filter: 4h HMA(50) for major trend alignment
- Trend: EMA(8)/EMA(21) crossover on 1h
- Entry: RSI(14) pullback (RSI < 45 in uptrend, RSI > 55 in downtrend)
- Volume: Volume > 1.2x 20-period average for confirmation
- Stoploss: 2.5*ATR(14) trailing
- Take profit: Reduce to half at 2.5R profit
- Position sizing: 0.25 base, 0.30 max (discrete levels)

Why this should beat current best:
- RSI pullback entries have better risk/reward than breakout entries
- 4h HMA filter removes counter-trend trades (major drawdown source)
- Volume confirmation filters low-conviction signals
- Conservative sizing (0.25-0.30) controls drawdown during crypto crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_rsi_pullback_4hhtf_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_ema(close, span):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=span, adjust=False, min_periods=span).mean()
    return ema.values


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
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period - 1, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


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


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong confirmation
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter (major trend direction)
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        major_trend = 1 if price_above_4h_hma else -1
        
        # EMA crossover signals
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # EMA crossover detection (fast crosses above/below slow)
        ema_cross_long = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_short = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # RSI pullback conditions
        rsi_pullback_long = rsi[i] < 45  # RSI dipped in uptrend
        rsi_pullback_short = rsi[i] > 55  # RSI rose in downtrend
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # Calculate position size based on volume strength
        volume_multiplier = min(1.0 + (volume[i] / vol_ma[i] - 1.2) / 2, 1.2)
        position_size = min(MAX_SIZE, max(BASE_SIZE, BASE_SIZE * volume_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: EMA cross + RSI pullback + major trend up + volume confirmed
        if (ema_cross_long and rsi_pullback_long and major_trend == 1 and volume_confirmed):
            target_signal = position_size
        
        # Short entry: EMA cross + RSI pullback + major trend down + volume confirmed
        elif (ema_cross_short and rsi_pullback_short and major_trend == -1 and volume_confirmed):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if low[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if high[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if high[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if low[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
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
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if EMA reverses OR major trend breaks
                ema_reversal_long = ema_fast[i] < ema_slow[i]
                ema_reversal_short = ema_fast[i] > ema_slow[i]
                hma_alignment_broken = (position_side == 1 and major_trend == -1) or \
                                       (position_side == -1 and major_trend == 1)
                
                if ema_reversal_long or ema_reversal_short or hma_alignment_broken:
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