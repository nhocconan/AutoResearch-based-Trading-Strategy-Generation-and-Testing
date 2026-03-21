#!/usr/bin/env python3
"""
EXPERIMENT #003 - Donchian Breakout + HTF Trend + Volume Filter (1h primary, 4h HTF)
====================================================================================
Hypothesis: Donchian breakouts (20-period high/low) capture trend momentum effectively
on 1h timeframe. Adding 4h HMA(21) as trend filter reduces false breakouts against
major trend. Volume confirmation (above 20-period average) ensures genuine breakouts
with participation. This differs from previous HMA/KAMA crossover strategies by using
pure price breakouts rather than MA crossovers, which should reduce lag and capture
stronger momentum moves.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filter: 4h HMA(21) for major trend direction
- Entry: Donchian(20) breakout - price breaks 20-period high/low
- Volume filter: Current volume > 1.2 * 20-period avg volume
- Exit: Supertrend(10,3) reversal OR 2.5*ATR trailing stop
- Position sizing: 0.25-0.30 discrete levels with ATR-based risk adjustment
- Take profit: Reduce to half at 2R profit, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_volume_htf_1h_4h_v1"
timeframe = "1h"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, n):
        if close[i - 1] <= supertrend[i - 1]:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
    
    return supertrend, direction


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 1h indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = BASE_SIZE / 2  # For take profit reduction
    MAX_SIZE = 0.35  # Maximum position size
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 1.0  # ATR at entry for R calculation
    profit_target_hit = False
    
    min_period = 50  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or 
            np.isnan(st_direction[i]) or np.isnan(vol_sma[i]) or 
            atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter
        htf_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Volume filter: current volume > 1.2 * 20-period average
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # Supertrend confirmation
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Donchian breakout + HTF bullish + Volume confirmed + Supertrend bullish
        if breakout_long and htf_trend == 1 and volume_confirmed and st_bullish:
            # Adjust size based on ATR (smaller position when volatility is high)
            atr_pct = atr[i] / close[i]
            size_multiplier = min(1.5, 0.02 / (atr_pct + 0.001))  # Target 2% risk
            adjusted_size = min(MAX_SIZE, BASE_SIZE * size_multiplier)
            target_signal = adjusted_size
        
        # Short entry: Donchian breakout + HTF bearish + Volume confirmed + Supertrend bearish
        elif breakout_short and htf_trend == -1 and volume_confirmed and st_bearish:
            atr_pct = atr[i] / close[i]
            size_multiplier = min(1.5, 0.02 / (atr_pct + 0.001))
            adjusted_size = min(MAX_SIZE, BASE_SIZE * size_multiplier)
            target_signal = -adjusted_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if low[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
                
                # Check supertrend reversal
                if st_direction[i] == -1:
                    trend_reversal = True
                    
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if high[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
                
                # Check supertrend reversal
                if st_direction[i] == 1:
                    trend_reversal = True
        
        # Apply signals based on conditions
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 1.0
            profit_target_hit = False
            
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            
        elif trend_reversal:
            # Supertrend reversed, exit position
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 1.0
            profit_target_hit = False
            
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
                # Maintain existing position
                current_size = BASE_SIZE if not profit_target_hit else HALF_SIZE
                signals[i] = current_size * position_side
            else:
                signals[i] = 0.0
    
    return signals