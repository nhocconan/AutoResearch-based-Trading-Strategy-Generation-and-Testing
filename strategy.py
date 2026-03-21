#!/usr/bin/env python3
"""
EXPERIMENT #034 - Donchian Breakout + Triple HTF Trend Filter (4h primary, 1d/1w HTF)
=====================================================================================
Hypothesis: Donchian Channel breakouts work best when aligned with multiple timeframe
trends. Using 4h for entries, 1d HMA(50) for intermediate trend, and 1w HMA(50) for
major trend creates a strong filter against false breakouts. Volume confirmation
ensures breakouts have participation. This differs from previous attempts by using
Donchian instead of Supertrend/RSI and adding weekly trend confirmation.

Key features:
- Primary TF: 4h (Donchian breakout signals)
- HTF filter 1: 1d HMA(50) for intermediate trend
- HTF filter 2: 1w HMA(50) for major trend (triple alignment)
- Breakout: Donchian(20) high/low break with volume confirmation
- Volume filter: breakout volume > 1.5x 20-period average
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25 discrete levels (conservative)
- Take profit: Reduce to half at 2R, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_triple_htf_volume_4h_1d_1w_v1"
timeframe = "4h"
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
    """Calculate Donchian Channel (highest high, lowest low over period)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    # Fill initial values with NaN
    upper[:period - 1] = np.nan
    lower[:period - 1] = np.nan
    
    return upper, lower


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = change / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = close[i]
    
    return kama


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMAs
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF data to 4h timeframe (Rule 2 - no manual mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    atr = calculate_atr(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    kama = calculate_kama(close, 10, 2, 30)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
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
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(kama[i]) or
            atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend filter
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # Volume confirmation
        volume_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirmed = volume_ratio > 1.5  # Breakout volume > 1.5x average
        
        # KAMA trend filter (adaptive trend)
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long breakout: Price breaks Donchian upper + all trends bullish + volume confirmed
        if (close[i] > donchian_upper[i - 1] and  # Break above previous upper
            daily_trend == 1 and 
            weekly_trend == 1 and 
            kama_trend == 1 and
            volume_confirmed):
            target_signal = SIZE
        
        # Short breakout: Price breaks Donchian lower + all trends bearish + volume confirmed
        elif (close[i] < donchian_lower[i - 1] and  # Break below previous lower
              daily_trend == -1 and 
              weekly_trend == -1 and 
              kama_trend == -1 and
              volume_confirmed):
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, where R = 2*ATR)
                if not profit_target_hit:
                    r_distance = 2.0 * entry_atr
                    if close[i] >= entry_price + 2.0 * r_distance:  # 2R = 4*entry_ATR
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
                    r_distance = 2.0 * entry_atr
                    if close[i] <= entry_price - 2.0 * r_distance:  # 2R profit
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
                if position_side == 1 and (daily_trend == -1 or weekly_trend == -1 or kama_trend == -1):
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                elif position_side == -1 and (daily_trend == 1 or weekly_trend == 1 or kama_trend == 1):
                    # Trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals