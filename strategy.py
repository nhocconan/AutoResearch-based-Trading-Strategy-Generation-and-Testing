#!/usr/bin/env python3
"""
Experiment #058: 4h KAMA Adaptive Trend with Daily/Weekly HMA Filter + Choppiness Regime
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than EMA/HMA.
In ranging markets (2025 bear/range), KAMA reduces whipsaws by flattening when ER is low.
Combine with 1d HMA for intermediate trend, 1w HMA for major trend bias.
Add Choppiness Index to only trade when CHOP < 50 (trending regime, not ranging).
This differs from #052 (Supertrend) by using adaptive MA instead of ATR-based trend.
Position sizing: 0.25 entry, 0.125 at 2R profit, stoploss at 2.5*ATR.
Target: Beat Sharpe=0.499 from #047 (12h Supertrend) with fewer but higher quality trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_weekly_hma_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise using Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER = trending (fast SC), Low ER = ranging (slow SC)
    """
    n = len(close)
    close = np.array(close)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        er[i] = signal / noise if noise > 0 else 0
    
    # Calculate Smoothing Constant
    fast_sc_val = 2.0 / (fast_sc + 1)
    slow_sc_val = 2.0 / (slow_sc + 1)
    sc = er ** 2 * (fast_sc_val - slow_sc_val) + slow_sc_val
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j - 1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j - 1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    kama_slow = calculate_kama(close, er_period=20, fast_sc=2, slow_sc=30)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # Weekly trend filter (major bias)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Daily trend filter (intermediate)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # Choppiness regime filter (only trade when trending)
        trending_regime = chop[i] < 50  # Below 50 = trending, above = ranging
        
        # KAMA crossover signals
        kama_cross_long = (i > 0) and (kama_fast[i] > kama_slow[i]) and (kama_fast[i-1] <= kama_slow[i-1])
        kama_cross_short = (i > 0) and (kama_fast[i] < kama_slow[i]) and (kama_fast[i-1] >= kama_slow[i-1])
        
        # KAMA trend alignment
        kama_trend_long = kama_fast[i] > kama_slow[i] and close[i] > kama_fast[i]
        kama_trend_short = kama_fast[i] < kama_slow[i] and close[i] < kama_fast[i]
        
        # RSI confirmation (avoid overbought/oversold entries)
        rsi_ok_long = 30 < rsi[i] < 70
        rsi_ok_short = 30 < rsi[i] < 70
        
        new_signal = 0.0
        
        # LONG ENTRY: KAMA cross + Weekly bullish + Daily bullish + Trending regime
        if kama_cross_long and weekly_bullish and daily_bullish and trending_regime and rsi_ok_long:
            new_signal = SIZE_ENTRY
        elif kama_trend_long and weekly_bullish and daily_bullish and trending_regime:
            # Continue position if already in trend
            if position_side == 1:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: KAMA cross + Weekly bearish + Daily bearish + Trending regime
        if kama_cross_short and weekly_bearish and daily_bearish and trending_regime and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        elif kama_trend_short and weekly_bearish and daily_bearish and trending_regime:
            # Continue position if already in trend
            if position_side == -1:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.5 * atr[int(np.where(np.arange(n) == np.argmin(np.abs(close - entry_price[:i+1])))[0][0])] if i > 0 else atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.5 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals