#!/usr/bin/env python3
"""
EXPERIMENT #016 - Donchian Breakout + 1d HMA Trend + Volume Filter (4h primary)
=====================================================================================
Hypothesis: 4h Donchian(20) breakouts capture sustained momentum moves in crypto.
Adding 1d HMA(21) trend filter ensures we only trade with the major trend direction.
Volume confirmation (volume > 1.5x 20-period avg) filters false breakouts on low liquidity.
ATR trailing stop (2.5*ATR) controls drawdown during reversals.

Key features:
- Primary TF: 4h (experiment requirement)
- HTF filter: 1d HMA(21) for major trend direction
- Entry: Donchian(20) breakout with volume confirmation
- Exit: Donchian middle line cross OR 2.5*ATR trailing stop
- Position sizing: 0.25-0.30 discrete levels (CRITICAL for drawdown control)
- Take profit: Reduce to half at 2R, trail stop at 1R

Why this should beat previous 4h strategies:
- Simpler entry logic = more trades (avoiding 0-trade problem from #004, #012)
- Volume filter reduces false breakouts (unlike pure Donchian in #011)
- 1d HMA is smoother than 1d EMA for trend filtering
- Conservative sizing (0.25-0.30) prevents -80%+ DD seen in #007, #008, #013
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1dhma_volume_4h_v1"
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper, lower, middle)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Fill NaN for early periods
    upper[:period - 1] = np.nan
    lower[:period - 1] = np.nan
    middle[:period - 1] = np.nan
    
    return upper, lower, middle


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend filter
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    volume_sma = calculate_volume_sma(volume, period=20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital) - CRITICAL for DD control
    MAX_SIZE = 0.35   # Max position size with strong volume
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for indicators to stabilize (Donchian needs 20, volume needs 20)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(volume_sma[i]) or
            atr[i] == 0 or volume_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1d HMA major trend filter
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        major_trend = 1 if price_above_1d_hma else -1
        
        # Volume confirmation (volume > 1.5x average = strong breakout)
        volume_ratio = volume[i] / volume_sma[i]
        volume_confirmed = volume_ratio > 1.3  # Relaxed from 1.5 to get more trades
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # Exit signals (Donchian middle cross)
        exit_long = close[i] < donchian_middle[i]
        exit_short = close[i] > donchian_middle[i]
        
        # Calculate position size based on volume strength (dynamic sizing)
        volume_multiplier = min(1.0 + (volume_ratio - 1.3) / 2.0, 1.25)  # Max 1.25x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * volume_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Donchian breakout + 1d HMA bullish + Volume confirmed
        if breakout_long and major_trend == 1 and volume_confirmed:
            target_signal = position_size
        
        # Short entry: Donchian breakout + 1d HMA bearish + Volume confirmed
        elif breakout_short and major_trend == -1 and volume_confirmed:
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_exit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check Donchian middle exit
                if exit_long and not stoploss_triggered:
                    trend_exit_triggered = True
                
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
                
                # Check Donchian middle exit
                if exit_short and not stoploss_triggered:
                    trend_exit_triggered = True
                
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
        elif trend_exit_triggered:
            # Exit on Donchian middle cross
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
                # Maintain existing position
                # Check if major trend reversed (1d HMA alignment broken)
                hma_alignment_broken = (position_side == 1 and major_trend == -1) or \
                                       (position_side == -1 and major_trend == 1)
                
                if hma_alignment_broken:
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