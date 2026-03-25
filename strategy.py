#!/usr/bin/env python3
"""
Experiment #1175: 6h Primary + 12h/1d HTF — Donchian Breakout + ADX Trend Filter

Hypothesis: After analyzing 970+ failed strategies, the winning pattern is SIMPLE trend-following
with momentum confirmation. Mean reversion (CRSI, Choppiness) consistently fails on BTC/ETH.

This strategy uses:
1. Donchian Channel(20) breakout for entry timing - captures momentum moves
2. 12h HMA(21) for trend direction filter - only trade with HTF trend
3. ADX(14) > 20 filter - ensures we're trading in trending markets, not chop
4. 1d HMA(21) for position sizing - stronger trend = larger position
5. ATR(14) 2.5x trailing stop - protects gains and limits drawdown

Why 6h Donchian might work:
- 6h is the "goldilocks" timeframe: slower than 4h (less noise), faster than 12h (more trades)
- Donchian(20) = ~5 days of data on 6h - captures multi-day momentum moves
- ADX filter prevents the #1 killer: trading in choppy/range markets
- Target: 30-50 trades/year (fee-friendly, avoids 0-trade failure)

Entry Logic:
- LONG: Price > 12h_HMA AND Donchian(20) high break AND ADX(14) > 20
- SHORT: Price < 12h_HMA AND Donchian(20) low break AND ADX(14) > 20
- Strong signal: Also aligned with 1d_HMA (SIZE_STRONG = 0.30)
- Base signal: Only 12h_HMA alignment (SIZE_BASE = 0.25)

Exit Logic:
- Trailing stop: 2.5x ATR from highest/lowest since entry
- Signal flip: When price crosses back through 12h_HMA

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_adx_trend_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    smooth_period = period * 2
    for i in range(smooth_period - 1, n):
        if i == smooth_period - 1:
            plus_tr = np.sum(tr[1:smooth_period])
            plus_dm_sum = np.sum(plus_dm[1:smooth_period])
            minus_dm_sum = np.sum(minus_dm[1:smooth_period])
        else:
            plus_tr = pd.Series(tr[:i+1]).ewm(span=smooth_period, min_periods=smooth_period, adjust=False).mean().iloc[-1] * smooth_period
            plus_dm_sum = pd.Series(plus_dm[:i+1]).ewm(span=smooth_period, min_periods=smooth_period, adjust=False).mean().iloc[-1] * smooth_period
            minus_dm_sum = pd.Series(minus_dm[:i+1]).ewm(span=smooth_period, min_periods=smooth_period, adjust=False).mean().iloc[-1] * smooth_period
        
        if plus_tr > 0:
            plus_di[i] = 100.0 * plus_dm_sum / plus_tr
        if plus_tr > 0:
            minus_di[i] = 100.0 * minus_dm_sum / plus_tr
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(smooth_period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:smooth_period + period - 1] = np.nan
    return adx

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
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
    
    # Track Donchian breaks
    prev_upper = np.nan
    prev_lower = np.nan
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]):
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
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (12h HMA) ===
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # 1d HMA for additional confirmation (position sizing)
        hma_1d_valid = not np.isnan(hma_1d_aligned[i])
        price_above_1d = hma_1d_valid and close[i] > hma_1d_aligned[i]
        price_below_1d = hma_1d_valid and close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        adx = adx_14[i]
        is_trending = adx > 20.0  # ADX > 20 = trending market
        
        # === DONCHIAN BREAKOUT DETECTION ===
        upper_break = False
        lower_break = False
        
        if not np.isnan(prev_upper) and not np.isnan(prev_lower):
            # Break above previous Donchian high
            if high[i] > prev_upper and close[i] > prev_upper:
                upper_break = True
            # Break below previous Donchian low
            if low[i] < prev_lower and close[i] < prev_lower:
                lower_break = True
        
        prev_upper = donchian_upper[i]
        prev_lower = donchian_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 12h trend up + Donchian breakout + ADX confirms trend
        if price_above_12h and upper_break and is_trending:
            if price_above_1d:
                desired_signal = SIZE_STRONG  # Strong trend alignment (12h + 1d)
            else:
                desired_signal = SIZE_BASE  # Basic uptrend breakout
        
        # SHORT: 12h trend down + Donchian breakout + ADX confirms trend
        elif price_below_12h and lower_break and is_trending:
            if price_below_1d:
                desired_signal = -SIZE_STRONG  # Strong trend alignment (12h + 1d)
            else:
                desired_signal = -SIZE_BASE  # Basic downtrend breakout
        
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
                entry_atr = atr_14[i]
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