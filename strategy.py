#!/usr/bin/env python3
"""
Experiment #1034: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX + RSI Pullback

Hypothesis: After analyzing 749+ failed strategies, the pattern is clear:
- Complex regime switching (Fisher+Choppiness+Vol) creates TOO MANY filters = 0 trades
- KAMA (Kaufman Adaptive Moving Average) naturally adapts to market efficiency
  - Fast in trends, slow in chop — no need for explicit regime detection
- ADX>20 (not 40!) filters weak trends without being too restrictive
- RSI 40/60 thresholds (not 30/70) generate MORE entry signals
- 12h KAMA for trend direction, 1d KAMA for bias — asymmetric for bear market

Why this should work:
1. KAMA ER (Efficiency Ratio) adapts automatically — no manual regime switch needed
2. Lower ADX threshold (20 vs 40) = more valid trend signals
3. RSI 40/60 = more frequent entries than 30/70 extremes
4. Simpler logic = fewer conditions that can all fail simultaneously
5. 4h timeframe targets 30-60 trades/year — enough for statistical significance

Critical fixes from failed experiments:
- REMOVED Choppiness Index (too many false negatives in transition periods)
- REMOVED Vol Spike filter (misses gradual trends, only catches panic)
- REMOVED Fisher Transform (complex, similar to RSI but less intuitive)
- LOWERED ADX threshold from 40 to 20 (ADX>40 rarely occurs)
- RELAXED RSI from 30/70 to 40/60 (more entry opportunities)
- ADDED explicit "hold position" logic to avoid premature exits

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_12h1d_pullback_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast SC - slow SC) + slow SC)^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 20 = trending, ADX < 20 = ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method (EMA-like)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(period * 2, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h KAMA10 for medium-term trend
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate and align 1d KAMA10 for long-term trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === MACRO TREND (HTF KAMA10) ===
        # Asymmetric: easier to long (12h), harder to short (1d) for bear bias
        medium_bull = close[i] > kama_12h_aligned[i]
        medium_bear = close[i] < kama_12h_aligned[i]
        long_bull = close[i] > kama_1d_aligned[i]
        long_bear = close[i] < kama_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_4h[i] > 20  # Trending (lowered from 40)
        trend_weak = adx_4h[i] <= 20   # Ranging
        
        # === DIRECTIONAL BIAS (DI) ===
        di_bullish = plus_di_4h[i] > minus_di_4h[i]
        di_bearish = minus_di_4h[i] > plus_di_4h[i]
        
        # === RSI ENTRY SIGNALS (relaxed thresholds) ===
        rsi_oversold = rsi_4h[i] < 40  # Lowered from 30
        rsi_overbought = rsi_4h[i] > 60  # Lowered from 70
        rsi_neutral = 40 <= rsi_4h[i] <= 60
        
        # === KAMA PULLBACK ENTRY ===
        # Long: price pulls back to KAMA in uptrend
        kama_pullback_long = close[i] < kama_4h[i] * 1.002 and close[i] > kama_4h[i] * 0.995
        # Short: price rallies to KAMA in downtrend
        kama_pullback_short = close[i] > kama_4h[i] * 0.998 and close[i] < kama_4h[i] * 1.005
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Condition 1: Trending bullish with RSI pullback
        if trend_strong and medium_bull and di_bullish:
            if rsi_oversold or (rsi_neutral and kama_pullback_long):
                desired_signal = BASE_SIZE
        # Condition 2: Ranging with mean reversion (easier entry)
        elif trend_weak and medium_bull:
            if rsi_oversold:
                desired_signal = REDUCED_SIZE
        # Condition 3: Long-term bullish (1d) — highest conviction
        elif long_bull and medium_bull:
            if rsi_4h[i] < 50:  # Any dip in strong uptrend
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRIES ===
        # Condition 1: Trending bearish with RSI rally
        if trend_strong and long_bear and di_bearish:
            if rsi_overbought or (rsi_neutral and kama_pullback_short):
                desired_signal = -BASE_SIZE
        # Condition 2: Ranging with mean reversion (easier entry)
        elif trend_weak and long_bear:
            if rsi_overbought:
                desired_signal = -REDUCED_SIZE
        # Condition 3: Long-term bearish (1d) — highest conviction
        elif long_bear and medium_bear:
            if rsi_4h[i] > 50:  # Any rally in strong downtrend
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        # This is CRITICAL to avoid churning and generate fewer but better trades
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if medium trend intact and RSI not extreme
                if medium_bull and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if long-term bearish and RSI not extreme
                if long_bear and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if medium trend reverses AND RSI overbought
            if not medium_bull and rsi_4h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if long-term trend reverses AND RSI oversold
            if not long_bear and rsi_4h[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals