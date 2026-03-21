#!/usr/bin/env python3
"""
EXPERIMENT #012 - Donchian Breakout with Weekly Trend + Volume Filter (1d)
==========================================================================
Hypothesis: Daily Donchian breakouts (20-day high/low) capture trend continuations
when aligned with weekly HMA(50) trend direction. Volume confirmation filters
false breakouts. ATR trailing stop protects against reversals. This differs from
previous KAMA/RSI approaches by using pure price breakout logic with volume validation.

Key features:
- Primary TF: 1d (daily candles)
- HTF filter: 1w HMA(50) for major trend direction
- Entry: Donchian(20) breakout + volume > 20-day average
- Filter: Weekly trend must align with breakout direction
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_volume_weekly_1d_v1"
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-day high/low)"""
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Volume moving average
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for weekly HMA and indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_sma[i]) or np.isnan(rsi[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # Volume confirmation (must be above 20-day average)
        volume_confirmed = volume[i] > volume_sma[i]
        
        # Donchian breakout detection
        breakout_signal = 0
        if i > 0:
            # Long breakout: price crosses above Donchian upper
            if close[i] > donchian_upper[i] and close[i - 1] <= donchian_upper[i - 1]:
                breakout_signal = 1
            # Short breakout: price crosses below Donchian lower
            elif close[i] < donchian_lower[i] and close[i - 1] >= donchian_lower[i - 1]:
                breakout_signal = -1
        
        # RSI filter (avoid extreme overbought/oversold entries)
        rsi_valid = 25 < rsi[i] < 75
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if breakout_signal != 0:
            # Breakout must align with weekly trend
            if breakout_signal == weekly_trend and volume_confirmed and rsi_valid:
                target_signal = SIZE * breakout_signal
        
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
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    profit_target = entry_price + 2.0 * (entry_price - (entry_price - 2.5 * atr[i]))
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR (since stop is 2.5*ATR)
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
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            # Trail stop tighter after TP (1R from highest/lowest)
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals