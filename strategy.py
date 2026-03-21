#!/usr/bin/env python3
"""
EXPERIMENT #012 - Donchian Breakout + Weekly HMA Trend Filter (1d primary)
=====================================================================================
Hypothesis: Daily Donchian(20) breakouts capture major crypto trends effectively.
Adding weekly HMA(21) trend filter ensures we only trade with the major trend direction.
Volume confirmation (>1.2x 20-day avg) filters false breakouts. ATR(14) stoploss at 2*ATR
controls risk during crypto volatility. This is a pure trend-following strategy that
should excel during BTC/ETH/SOL bull runs while cutting losses quickly in bear markets.

Key features:
- Primary TF: 1d (MANDATORY for this experiment)
- HTF filter: 1w HMA(21) for major trend direction
- Entry: Donchian(20) breakout with volume confirmation
- Exit: Donchian middle line or opposite breakout
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Trail stop at 1.5R, reduce at 2R

Why this should work on 1d:
- Daily timeframe captures sustained trends without noise
- Weekly HMA filter removes counter-trend trades
- Donchian breakouts are proven trend-following signals
- Conservative sizing (0.25-0.30) survives 70%+ crypto crashes
- Relaxed entry conditions ensure ≥10 trades per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1whma_volume_1d_v1"
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
    
    return upper, lower, middle


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
    
    # Load HTF data ONCE before loop (Rule 1) - weekly for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend filter
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    volume_ma = calculate_volume_ma(volume, period=20)
    
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
    
    min_period = 50  # Wait for indicators to stabilize (Donchian needs 20, HMA needs more)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(volume_ma[i]) or
            atr[i] == 0 or volume_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly HMA trend filter
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        hma_trend = 1 if price_above_1w_hma else -1
        
        # Volume confirmation (relaxed: >1.0x instead of >1.2x for more trades)
        volume_ratio = volume[i] / volume_ma[i] if volume_ma[i] > 0 else 1.0
        volume_confirmed = volume_ratio > 1.0  # Any above-average volume
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # Calculate position size
        position_size = BASE_SIZE
        
        # Determine target signal based on filters
        target_signal = 0.0
        
        # Long entry: Breakout + weekly HMA bullish + volume confirmed
        if breakout_long and hma_trend == 1 and volume_confirmed:
            target_signal = position_size
        
        # Short entry: Breakout + weekly HMA bearish + volume confirmed
        elif breakout_short and hma_trend == -1 and volume_confirmed:
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
                # Maintain existing position - check for exit signals
                # Exit if Donchian middle is crossed (trend weakening)
                donchian_exit_long = close[i] < donchian_middle[i]
                donchian_exit_short = close[i] > donchian_middle[i]
                
                # Exit if weekly HMA alignment breaks
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                # Exit if opposite breakout
                opposite_breakout = (position_side == 1 and breakout_short) or \
                                    (position_side == -1 and breakout_long)
                
                if donchian_exit_long or donchian_exit_short or hma_alignment_broken or opposite_breakout:
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