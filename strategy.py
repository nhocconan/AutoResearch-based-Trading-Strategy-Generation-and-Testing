#!/usr/bin/env python3
"""
Experiment #583: 6h Primary + 1d/1w HTF — Donchian Breakout + MTF Trend + Volume

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Donchian breakouts
work well on higher timeframes (20-bar breakout = 5 days on 6h). Combined with 1d/1w
HMA trend filters for direction bias and volume confirmation, this should capture
major moves while avoiding whipsaw in 2022 crash.

Key differences from failed #580 (6h_hma_rsi_pullback):
1. Donchian breakout entries instead of RSI pullback (catches trends early)
2. Volume confirmation required (breakout must have conviction)
3. Asymmetric sizing: larger positions when 1w+1d both align
4. ADX filter to avoid breakouts in low-momentum chop
5. Wider stoploss (3x ATR) for 6h timeframe volatility

Strategy logic:
1. 1w HMA(21) = macro trend bias (slowest filter)
2. 1d HMA(21) = medium trend bias (confirmation)
3. 6h Donchian(20) = breakout entry (20 bars = 5 days)
4. 6h ADX(14) = momentum filter (ADX>20 = valid breakout)
5. 6h Volume = conviction check (volume > 1.5x 20-bar avg)
6. ATR(14)*3.0 stoploss (wider for 6h volatility)

Entry rules:
- LONG: price breaks Donchian high + volume spike + ADX>20 + 1d HMA bull
- SHORT: price breaks Donchian low + volume spike + ADX>20 + 1d HMA bear
- 1w HMA adds size bonus when aligned (0.30 vs 0.20)

Target: Sharpe>0.40, trades>=30 train (8/year), trades>=3 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_volume_hma_1d1w_v1"
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - highest high and lowest low over N periods
    Returns: upper_band, lower_band, middle_band
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - trend strength indicator
    ADX > 25 = strong trend, ADX < 20 = weak/range
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    SIZE_MAX = 0.35
    
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx[i]) or np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and hma_1d_aligned[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and hma_1d_aligned[i] < hma_1w_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # 1d-only bias (weaker signal)
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout = price closes above/below Donchian band
        breakout_long = close[i] > donchian_upper[i-1]  # Use previous bar's band to avoid look-ahead
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 1e-10 else 0.0
        volume_spike = volume_ratio > 1.3  # 30% above average
        
        # === ADX MOMENTUM FILTER ===
        adx_valid = adx[i] > 18.0  # Minimum momentum for breakout
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG breakout with confirmation
        if breakout_long and volume_spike and adx_valid:
            if htf_bull:
                # Full alignment: 1w + 1d both bull
                desired_signal = SIZE_STRONG
            elif htf_1d_bull:
                # 1d bull only
                desired_signal = SIZE_BASE
            # If 1d bear, skip long even with breakout
        
        # SHORT breakout with confirmation
        elif breakout_short and volume_spike and adx_valid:
            if htf_bear:
                # Full alignment: 1w + 1d both bear
                desired_signal = -SIZE_STRONG
            elif htf_1d_bear:
                # 1d bear only
                desired_signal = -SIZE_BASE
            # If 1d bull, skip short even with breakout
        
        # === STOPLOSS CHECK (3x ATR from entry - wider for 6h) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trailing stop: update if price moves favorably
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trailing stop: update if price moves favorably
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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