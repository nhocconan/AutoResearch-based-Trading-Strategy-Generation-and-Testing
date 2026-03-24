#!/usr/bin/env python3
"""
Experiment #697: 15m Primary + 4h/12h HTF — Camarilla Pivot + RSI + Choppiness Regime

Hypothesis: 15m timeframe can work with VERY selective entries using 3+ confluence:
1. 4h HMA(21) trend direction (primary bias)
2. 12h Choppiness Index regime filter (trend vs range)
3. 15m RSI(7) extreme for entry timing (more sensitive than RSI(14))
4. Camarilla R3/S3 levels from 1d HTF for mean reversion targets
5. Session filter: 00-12 UTC only (London/NY overlap for crypto)

Key innovations:
- Camarilla pivot levels provide natural support/resistance for mean reversion
- Choppiness Index prevents trend strategies in ranging markets
- RSI(7) catches faster pullbacks than RSI(14) on 15m
- Session filter reduces low-volume whipsaw trades
- Discrete sizing: 0.15 base, 0.25 strong (smaller for 15m frequency)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Trade frequency: 40-100 trades/year (CRITICAL for 15m)
Timeframe: 15m
Size: 0.15-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_rsi_chop_4h12d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies trending vs ranging markets
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_camarilla_pivots(open_price, high, low, close):
    """
    Camarilla Pivot Levels - intraday support/resistance
    R3/R4 = resistance, S3/S4 = support
    Pivot = (H + L + C) / 3
    Range = H - L
    R3 = C + Range * 1.1/4, R4 = C + Range * 1.1/2
    S3 = C - Range * 1.1/4, S4 = C - Range * 1.1/2
    """
    pivot = (high + low + close) / 3.0
    price_range = high - low
    
    r4 = close + price_range * 1.1 / 2.0
    r3 = close + price_range * 1.1 / 4.0
    s3 = close - price_range * 1.1 / 4.0
    s4 = close - price_range * 1.1 / 2.0
    
    return pivot, r3, r4, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate 1d Camarilla pivots (for mean reversion targets)
    camarilla_1d = calculate_camarilla_pivots(
        df_1d['open'].values,
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    pivot_1d_raw = camarilla_1d[0]
    r3_1d_raw = camarilla_1d[1]
    s3_1d_raw = camarilla_1d[3]
    
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_raw)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    
    # Session filter: 00-12 UTC only (hours 0-11)
    # Extract hour from open_time (assuming Unix timestamp in milliseconds)
    open_time = prices["open_time"].values
    hours = ((open_time // 3600000) % 24).astype(int)
    session_filter = (hours >= 0) & (hours < 12)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        if not session_filter[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (12h Choppiness) ===
        chop_value = chop_12h_aligned[i]
        is_trending = chop_value < 45.0  # trending regime
        is_ranging = chop_value > 55.0   # ranging regime
        
        # === 15m RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        rsi_neutral = 35.0 <= rsi_7[i] <= 65.0
        
        # === CAMARILLA LEVEL POSITION ===
        near_s3 = abs(close[i] - s3_1d_aligned[i]) < atr[i] * 1.5
        near_r3 = abs(close[i] - r3_1d_aligned[i]) < atr[i] * 1.5
        above_pivot = close[i] > pivot_1d_aligned[i]
        below_pivot = close[i] < pivot_1d_aligned[i]
        
        # === 15m HMA SHORT-TERM TREND ===
    hma_15m_bull = hma_21[i] > hma_21[i-1] if i > 0 else False
        hma_15m_bear = hma_21[i] < hma_21[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + near S3 or above pivot
        # Confluence: HTF trend + RSI extreme + Camarilla level
        long_confluence = 0
        if htf_4h_bull:
            long_confluence += 1
        if rsi_oversold:
            long_confluence += 1
        if near_s3 or above_pivot:
            long_confluence += 1
        if hma_15m_bull:
            long_confluence += 1
        
        if long_confluence >= 3 and rsi_oversold:
            if near_s3 and is_ranging:
                # Mean reversion in range at S3
                desired_signal = SIZE_STRONG
            elif htf_4h_bull and is_trending:
                # Trend pullback
                desired_signal = SIZE_BASE
            elif long_confluence >= 3:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + RSI overbought + near R3 or below pivot
        short_confluence = 0
        if htf_4h_bear:
            short_confluence += 1
        if rsi_overbought:
            short_confluence += 1
        if near_r3 or below_pivot:
            short_confluence += 1
        if hma_15m_bear:
            short_confluence += 1
        
        if short_confluence >= 3 and rsi_overbought:
            if near_r3 and is_ranging:
                # Mean reversion in range at R3
                desired_signal = -SIZE_STRONG
            elif htf_4h_bear and is_trending:
                # Trend pullback
                desired_signal = -SIZE_BASE
            elif short_confluence >= 3:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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