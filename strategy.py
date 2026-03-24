#!/usr/bin/env python3
"""
Experiment #1092: 12h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Regime + RSI

Hypothesis: 12h timeframe with adaptive KAMA trend + Choppiness regime detection will
capture multi-day swings while avoiding whipsaws. KAMA adapts smoothing based on market
efficiency ratio (tight in trends, loose in ranges). Choppiness filter switches between
trend-following and mean-reversion modes. 1d HTF provides trend bias.

Key innovations:
1. KAMA(10) - Adaptive smoothing that reduces lag in trends, increases in noise
2. Choppiness Index(14) - Regime detection: >55 range, <45 trend
3. RSI(14) - Entry timing within regime
4. 1d KAMA - HTF trend bias filter (simpler than triple HMA)
5. Donchian(20) - Breakout confirmation for trend entries
6. ATR(14) 2.5x trailing stop

Entry conditions (LOOSE for trade generation - critical for 12h):
- LONG trend: CHOP<50 + price>KAMA_12h + HTF_bull + RSI>45
- LONG mean-rev: CHOP>55 + RSI<40 + HTF_bull
- SHORT trend: CHOP<50 + price<KAMA_12h + HTF_bear + RSI<55
- SHORT mean-rev: CHOP>55 + RSI>60 + HTF_bear

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_rsi_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman's Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=np.float64)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(close[i]) or np.isnan(close[i - period]):
            continue
        
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 1e-10:
            er = signal / noise
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=np.float64)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=np.float64)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=np.float64)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 12h indicators
    kama_12h = calculate_kama(close, period=10)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n, dtype=np.float64)
    
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Check if indicators are valid
        if np.isnan(kama_12h[i]) or np.isnan(kama_1d_aligned[i]) or \
           np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === HTF BIAS ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === TREND FILTER ===
        trend_bull = close[i] > kama_12h[i]
        trend_bear = close[i] < kama_12h[i]
        
        # === BREAKOUT SIGNALS ===
        breakout_long = close[i] >= donchian_upper[i] * 0.995
        breakout_short = close[i] <= donchian_lower[i] * 1.005
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        if is_trending:
            # Trend following mode - require HTF alignment
            if trend_bull and htf_bull and rsi[i] > 45.0:
                desired_signal = SIZE_BASE
                if breakout_long:
                    desired_signal = SIZE_STRONG
            elif trend_bear and htf_bear and rsi[i] < 55.0:
                desired_signal = -SIZE_BASE
                if breakout_short:
                    desired_signal = -SIZE_STRONG
        
        elif is_choppy:
            # Mean reversion mode - fade extremes with HTF bias
            if htf_bull and rsi[i] < 40.0:
                desired_signal = SIZE_BASE
            elif htf_bear and rsi[i] > 60.0:
                desired_signal = -SIZE_BASE
        else:
            # Neutral regime - use simpler signals
            if htf_bull and trend_bull and rsi[i] > 50.0:
                desired_signal = SIZE_BASE
            elif htf_bear and trend_bear and rsi[i] < 50.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals