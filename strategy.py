#!/usr/bin/env python3
"""
Experiment #1482: 12h Primary + 1d/1w HTF — Simplified HMA Trend Following

Hypothesis: After 1105 failed strategies, the clearest pattern is:
1. Higher timeframes (12h, 1d) work better than lower TF (4h and below)
2. Simple trend-following with HTF filter beats complex regime-switching
3. HMA (Hull MA) responds faster than EMA/KAMA with less lag
4. Dual HTF filter (1d + 1w) provides stronger trend confirmation

This strategy uses:
- 1w HMA for ultra-macro trend bias (strongest filter)
- 1d HMA for macro trend direction
- 12h HMA crossover for entry timing
- Donchian(20) breakout confirmation
- RSI(14) loose filter (35-65 range for sufficient trades)
- ATR(14)*2.5 trailing stoploss

Why 12h + 1d + 1w should work:
1. 12h = target 20-50 trades/year (minimal fee drag ~1-2.5%)
2. 1w HMA filter prevents trading against ultra-macro trend
3. 1d HMA confirms intermediate trend direction
4. HMA has less lag than EMA, catches trends earlier
5. Loose RSI filter ensures we get enough trades (>10 per symbol)
6. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Timeframe: 12h
HTF: 1d and 1w (call get_htf_data ONCE before loop!)
Position Size: 0.25-0.30 (discrete levels)
Target: 20-50 trades/year, Sharpe > 0.618 (beat current best), ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_donchian_rsi_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - faster response with less lag than EMA
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, window):
        if len(series) < window:
            return np.full(len(series), np.nan)
        weights = np.arange(1, window + 1)
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            if np.any(np.isnan(series[i - window + 1:i + 1])):
                continue
            result[i] = np.sum(series[i - window + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    close_series = pd.Series(close)
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    if wma_half is None or wma_full is None:
        return np.full(n, np.nan)
    
    # HMA calculation
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=9)  # Faster HMA for crossover
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - strongest filter ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) - direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === HMA CROSSOVER (faster signal) ===
        hma_cross_bull = hma_12h_fast[i] > hma_12h[i]
        hma_cross_bear = hma_12h_fast[i] < hma_12h[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI MOMENTUM - LOOSE bands for more trades (35-65) ===
        rsi_bullish = rsi[i] > 35.0
        rsi_bearish = rsi[i] < 65.0
        rsi_strong_bull = rsi[i] > 45.0
        rsi_strong_bear = rsi[i] < 55.0
        
        # === DESIRED SIGNAL - SIMPLIFIED TREND FOLLOWING ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + Daily bull + 12h bull + Breakout or HMA cross + RSI support
        if weekly_bull and daily_bull and hma_bull:
            if breakout_high and rsi_bullish:
                desired_signal = BASE_SIZE
            elif hma_cross_bull and rsi_strong_bull:
                desired_signal = BASE_SIZE * 0.85
            elif hma_bull and rsi[i] > 40.0:
                desired_signal = BASE_SIZE * 0.65  # Weaker signal
        
        # SHORT: Weekly bear + Daily bear + 12h bear + Breakout or HMA cross + RSI support
        elif weekly_bear and daily_bear and hma_bear:
            if breakout_low and rsi_bearish:
                desired_signal = -BASE_SIZE
            elif hma_cross_bear and rsi_strong_bear:
                desired_signal = -BASE_SIZE * 0.85
            elif hma_bear and rsi[i] < 60.0:
                desired_signal = -BASE_SIZE * 0.65  # Weaker signal
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.75:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.75
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.75:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.75
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals