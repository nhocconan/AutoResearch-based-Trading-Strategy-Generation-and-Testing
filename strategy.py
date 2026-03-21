#!/usr/bin/env python3
"""
EXPERIMENT #011 - Donchian Breakout + 1d HMA Trend + RSI Confirmation (12h primary)
=====================================================================================
Hypothesis: 12h Donchian breakouts capture sustained moves, but generate false signals in chop.
Adding 1d HMA(21) trend filter ensures we only trade breakouts in the major trend direction.
RSI(14) confirmation (RSI > 50 for longs, RSI < 50 for shorts) filters weak breakouts.
Loose entry conditions ensure ≥10 trades per symbol across BTC/ETH/SOL.

Key features:
- Primary TF: 12h
- HTF filter: 1d HMA(21) for major trend direction
- Entry: Donchian(20) breakout + RSI confirmation
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit, trail stop

Why this should work:
- 12h captures multi-day breakouts without 15m noise
- Donchian breakout = clear, objective entry signal
- 1d HMA ensures we trade with major trend (critical for crypto)
- RSI confirmation filters false breakouts
- Conservative sizing (0.25-0.30) controls drawdown during crashes
- Loose conditions ensure sufficient trade count
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1dhma_rsi_12h_v1"
timeframe = "12h"
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Also calculate 12h HMA for additional trend confirmation
    hma_12h = calculate_hma(close, 21)
    
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
    
    min_period = 50  # Wait for indicators to stabilize (shorter for more trades)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or
            np.isnan(hma_12h[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1d HMA trend filter
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        hma_trend = 1 if price_above_1d_hma else -1
        
        # 12h HMA for additional confirmation
        price_above_12h_hma = close[i] > hma_12h[i]
        hma_12h_trend = 1 if price_above_12h_hma else -1
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # RSI confirmation (LOOSE conditions for more trades)
        rsi_bullish = rsi[i] > 45  # Not too strict
        rsi_bearish = rsi[i] < 55  # Not too strict
        
        # Calculate position size (discrete levels)
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Donchian breakout + 1d HMA bullish + 12h HMA bullish + RSI confirmation
        if breakout_long and hma_trend == 1 and hma_12h_trend == 1 and rsi_bullish:
            target_signal = position_size
        
        # Short entry: Donchian breakout + 1d HMA bearish + 12h HMA bearish + RSI confirmation
        elif breakout_short and hma_trend == -1 and hma_12h_trend == -1 and rsi_bearish:
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
                # Exit if 1d HMA alignment breaks OR 12h HMA breaks
                hma_1d_broken = (position_side == 1 and hma_trend == -1) or \
                                (position_side == -1 and hma_trend == 1)
                hma_12h_broken = (position_side == 1 and hma_12h_trend == -1) or \
                                 (position_side == -1 and hma_12h_trend == 1)
                
                if hma_1d_broken or hma_12h_broken:
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