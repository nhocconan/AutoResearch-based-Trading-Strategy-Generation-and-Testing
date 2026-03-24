#!/usr/bin/env python3
"""
Experiment #431: 6h Primary + 1w/1d HTF — Donchian Breakout with Weekly Pivot Bias

Hypothesis: Previous 6h strategies failed due to either (a) too strict entries = 0 trades,
or (b) mean-reversion in trending markets. This uses MOMENTUM breakout logic instead.

Key innovations:
1. Donchian(20) breakout on 6h - catches multi-day momentum swings (proven on crypto)
2. Weekly pivot bias: only long breakouts when price > weekly pivot, short when < pivot
3. 1d ADX > 20 filter - avoids breakouts in choppy/range markets
4. RSI(14) confirmation - avoid breakouts at extreme RSI (>75 long, <25 short = exhaustion)
5. Volume confirmation optional (not required - caused 0 trades in past)

Why this differs from failed 6h strategies:
- NOT CRSI-based (#420, #422, #426 all failed with CRSI)
- NOT pure mean reversion (fails in strong trends like 2021, 2024)
- Uses Donchian breakouts (momentum) instead of HMA crossovers
- Weekly pivot gives stronger directional bias than HMA alone

Position sizing: 0.25 base, 0.30 when weekly + daily aligned
Stoploss: 2.5x ATR(14) from entry
Target: 40-80 trades/year (6h = ~3500 bars/year, need ~1-2% trade rate)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_breakout_weekly_pivot_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_pivot_points(high, low, close):
    """Standard Pivot Points (P, R1, S1, R2, S2)"""
    n = len(close)
    pivot = np.zeros(n)
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    r2 = np.zeros(n)
    s2 = np.zeros(n)
    
    pivot[:] = np.nan
    r1[:] = np.nan
    s1[:] = np.nan
    r2[:] = np.nan
    s2[:] = np.nan
    
    for i in range(1, n):
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        
        pivot[i] = (h + l + c) / 3.0
        r1[i] = 2.0 * pivot[i] - l
        s1[i] = 2.0 * pivot[i] - h
        r2[i] = pivot[i] + (h - l)
        s2[i] = pivot[i] - (h - l)
    
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    # Weekly HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Weekly pivot for directional bias
    pivot_1w, r1_1w, s1_1w, _, _ = calculate_pivot_points(
        df_1w['high'].values, 
        df_1w['low'].values, 
        df_1w['close'].values
    )
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Daily ADX for trend strength filter
    adx_1d_raw = calculate_adx(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Daily HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    adx_6h = calculate_adx(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Check HTF alignment
        if np.isnan(hma_1w_aligned[i]) or np.isnan(pivot_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_1d_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (Weekly) ===
        # Long bias: price > weekly pivot AND price > weekly HMA
        # Short bias: price < weekly pivot AND price < weekly HMA
        weekly_bull = close[i] > pivot_1w_aligned[i] and close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < pivot_1w_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # === HTF FILTER (Daily ADX) ===
        # Only take breakouts when daily ADX > 20 (trending market)
        daily_trending = adx_1d_aligned[i] > 20.0
        
        # === Daily HMA Confirmation ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h DONCHIAN BREAKOUT ===
        # Long: price breaks above Donchian upper (20-period high)
        # Short: price breaks below Donchian lower (20-period low)
        breakout_long = close[i] > donchian_upper[i-1]  # break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # break below previous lower
        
        # === 6h ADX Confirmation ===
        six_h_trending = adx_6h[i] > 18.0
        
        # === RSI Filter (avoid exhaustion) ===
        # Don't go long if RSI > 75 (overbought exhaustion)
        # Don't go short if RSI < 25 (oversold exhaustion)
        rsi_ok_long = rsi[i] < 75.0
        rsi_ok_short = rsi[i] > 25.0
        
        # === SMA Filter (long-term trend) ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + weekly bull + daily trending + RSI ok
        if breakout_long and weekly_bull and daily_trending and rsi_ok_long:
            # Strong signal if daily also bull
            if daily_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: Donchian breakout + weekly bear + daily trending + RSI ok
        elif breakout_short and weekly_bear and daily_trending and rsi_ok_short:
            # Strong signal if daily also bear
            if daily_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals