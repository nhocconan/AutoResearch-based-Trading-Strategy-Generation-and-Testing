#!/usr/bin/env python3
"""
EXPERIMENT #017 - MTF HMA Crossover + RSI + Daily Trend (4h+1d v1)
==================================================================================================
Hypothesis: Use 4h as PRIMARY timeframe (less noise than 1h/15m) + 1d trend filter.
This differs from current best by:
- 4h primary instead of 1h (fewer but higher quality signals)
- HMA crossover for entry timing (faster than SMA, smoother than EMA)
- Daily SMA(50) as major trend filter (stronger than 4h Supertrend alone)
- RSI pullback confirmation on 4h (not overbought/oversold at entry)

Why this should work:
- 4h timeframe reduces whipsaws from lower TFs while catching major moves
- Daily SMA(50) filters out counter-trend trades in major bear/bull markets
- HMA crossover provides faster entry than traditional MA crosses
- RSI pullback ensures we're not chasing extremes
- Discrete signal levels (0, ±0.25, ±0.35) reduce churn costs

Position sizing: MAX 0.35 (35% capital), typical 0.25
Stoploss: 2.5*ATR trailing stop
Take profit: Reduce to half at 2R, trail at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_crossover_rsi_daily_4h_1d_v1"
timeframe = "4h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < er_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period - 1] = close[er_period - 1]
    
    for i in range(er_period, n):
        kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 4h indicators for entry timing
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    
    # HMA crossover signals (fast/slow)
    hma_fast_4h = calculate_hma(close, period=16)
    hma_slow_4h = calculate_hma(close, period=48)
    
    # KAMA for trend confirmation
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Get 1d data using mtf_data helper for major trend filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # Daily SMA(50) for major trend direction
        sma_50_1d = calculate_sma(c_1d, period=50)
        
        # Align daily indicators to 4h timeframe
        sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
        
    except Exception:
        # Fallback if mtf_data fails
        sma_50_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.25
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Minimum bars for valid signals
    first_valid = max(200, 50 * 6, 48, 14 * 2)  # Need enough data for daily SMA
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_4h[i]) or np.isnan(rsi_4h[i]) or atr_4h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned daily values
        sma_50_daily = sma_50_1d_aligned[i] if i < len(sma_50_1d_aligned) else 0
        
        # Daily trend filter (price vs SMA50)
        daily_trend = 0
        if sma_50_daily > 0:
            if close[i] > sma_50_daily:
                daily_trend = 1  # Bullish
            elif close[i] < sma_50_daily:
                daily_trend = -1  # Bearish
        
        # 4h HMA crossover signal
        hma_crossover = 0
        if i > 0:
            if hma_fast_4h[i] > hma_slow_4h[i] and hma_fast_4h[i-1] <= hma_slow_4h[i-1]:
                hma_crossover = 1  # Bullish crossover
            elif hma_fast_4h[i] < hma_slow_4h[i] and hma_fast_4h[i-1] >= hma_slow_4h[i-1]:
                hma_crossover = -1  # Bearish crossover
        
        # KAMA trend confirmation
        kama_trend = 0
        if kama_4h[i] > 0:
            if close[i] > kama_4h[i]:
                kama_trend = 1
            elif close[i] < kama_4h[i]:
                kama_trend = -1
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_4h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_4h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_4h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_4h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_4h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_4h[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Check if trend reversed (daily SMA filter)
            if prev_side == 1 and daily_trend == -1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            elif prev_side == -1 and daily_trend == 1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: Daily trend + 4h HMA crossover + RSI pullback + KAMA confirmation
        price = close[i]
        
        # Long entry conditions
        if daily_trend == 1 and hma_crossover == 1 and kama_trend == 1:
            # RSI pullback (not overbought)
            if RSI_LONG_MIN <= rsi_4h[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_QUARTER
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # Short entry conditions
        elif daily_trend == -1 and hma_crossover == -1 and kama_trend == -1:
            # RSI pullback (not oversold)
            if RSI_SHORT_MIN <= rsi_4h[i] <= RSI_SHORT_MAX:
                signals[i] = -SIZE_QUARTER
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals