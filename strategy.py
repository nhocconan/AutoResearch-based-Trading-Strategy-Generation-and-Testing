#!/usr/bin/env python3
"""
EXPERIMENT #016 - KAMA Adaptive Trend + Volume Momentum + Daily Filter (4h)
===========================================================================
Hypothesis: KAMA (Kaufman's Adaptive Moving Average) adapts to market noise better
than HMA/EMA, providing cleaner trend signals on 4h timeframe. Combined with volume
momentum confirmation and daily HMA trend filter, this should capture sustained
trends while filtering choppy periods. Different from failed KAMA strategies by:
- Using 4h primary timeframe (not 12h/30m)
- Adding volume momentum ratio as entry confirmation
- Daily HMA filter instead of weekly
- Tighter ATR stoploss (2.0x instead of 2.5x)
- Discrete position sizing to reduce fee churn

Key features:
- Primary TF: 4h (four-hour candles)
- HTF filter: 1d HMA(50) for major trend direction
- Entry: KAMA(14) crossover + volume ratio > 1.5
- Filter: Daily trend must align with signal direction
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_volume_momentum_daily_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA)
    KAMA adapts to market noise using Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = 0.0
        for j in range(i - period + 1, i + 1):
            sum_changes += abs(close[j] - close[j - 1])
        er[i] = net_change / (sum_changes + 1e-10) if sum_changes > 0 else 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average"""
    volume_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    volume_ratio = volume / (volume_sma + 1e-10)
    return volume_ratio


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
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, period=14, fast=2, slow=30)
    kama_fast = calculate_kama(close, period=7, fast=2, slow=15)
    atr = calculate_atr(high, low, close, 14)
    volume_ratio = calculate_volume_ratio(volume, 20)
    rsi = calculate_rsi(close, 14)
    
    # KAMA crossover signal (fast crosses slow)
    kama_signal = np.zeros(n)
    for i in range(20, n):
        if not np.isnan(kama_fast[i]) and not np.isnan(kama[i]):
            if kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1]:
                kama_signal[i] = 1
            elif kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1]:
                kama_signal[i] = -1
    
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
    entry_atr = 0.0
    
    min_period = 100  # Wait for daily HMA and indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(kama_fast[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(rsi[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # Volume momentum confirmation (must be above 1.5x average)
        volume_confirmed = volume_ratio[i] > 1.5
        
        # RSI filter (avoid extreme overbought/oversold entries)
        rsi_valid = 30 < rsi[i] < 70
        
        # KAMA crossover signal
        kama_breakout = kama_signal[i]
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if kama_breakout != 0:
            # Signal must align with daily trend
            if kama_breakout == daily_trend and volume_confirmed and rsi_valid:
                target_signal = SIZE * kama_breakout
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    risk = entry_price - (entry_price - 2.0 * entry_atr)
                    if close[i] >= entry_price + 2.0 * risk:  # 2R profit
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    risk = (entry_price + 2.0 * entry_atr) - entry_price
                    if close[i] <= entry_price - 2.0 * risk:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            entry_atr = 0.0
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
                # Maintain existing position (or reverse if strong opposite signal)
                if target_signal != 0.0 and np.sign(target_signal) != position_side:
                    # Reverse position
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    entry_atr = atr[i]
                    profit_target_hit = False
                else:
                    # Keep current position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals