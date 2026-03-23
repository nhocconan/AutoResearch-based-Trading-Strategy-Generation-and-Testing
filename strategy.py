#!/usr/bin/env python3
"""
Experiment #1355: 1h Primary + 4h/1d HTF — Simplified Multi-TF Trend Following

Hypothesis: Based on #1352 (12h, Sharpe=0.571) success with SIMPLE Donchian+HMA+RSI,
and #1350 (1h regime) failure with over-filtering, this strategy uses:
1. 4h HMA(21) for primary trend bias (proven in #1345, #1354)
2. 1d HMA(21) for macro confirmation (soft filter)
3. 1h Donchian(20) breakout as trigger (momentum entry)
4. RSI(14) with MODERATE bands (35-65) — confirms without over-filtering
5. Choppiness Index as REGIME GATE — only trade when CHOP matches strategy type
6. ATR(14) trailing stop 2.5x — proven risk management
7. Position size 0.22 — conservative for 1h volatility

Key difference from failed #1348 (30m): Fewer trades via stricter confluence.
Key difference from failed #1350 (1h regime): No dual-regime switching, single mode.

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_hma_rsi_chop_4h1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
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
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP > 61.8 = ranging/choppy
    CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend filters
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d HMA) - soft filter ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === CHOPPY INDEX REGIME GATE ===
        # Only trade breakouts when trending (CHOP < 50)
        # Only trade mean-reversion when ranging (CHOP > 55)
        is_trending = chop[i] < 50.0
        is_ranging = chop[i] > 55.0
        
        # === RSI MOMENTUM (moderate bands for trade frequency) ===
        rsi_bull = rsi[i] > 35.0
        rsi_bear = rsi[i] < 65.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY paths (multiple to ensure trades)
        # Path 1: Donchian breakout + trending regime + 4h bull (primary)
        if breakout_long and is_trending and trend_bull:
            desired_signal = BASE_SIZE
        # Path 2: Price above both HTF HMAs + RSI confirmation (trend follow)
        elif close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i] and rsi_bull:
            desired_signal = BASE_SIZE * 0.5
        # Path 3: Ranging regime + RSI oversold + macro bull (mean revert long)
        elif is_ranging and rsi_oversold and macro_bull:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY paths (multiple to ensure trades)
        # Path 1: Donchian breakout + trending regime + 4h bear (primary)
        elif breakout_short and is_trending and trend_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Price below both HTF HMAs + RSI confirmation (trend follow)
        elif close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i] and rsi_bear:
            desired_signal = -BASE_SIZE * 0.5
        # Path 3: Ranging regime + RSI overbought + macro bear (mean revert short)
        elif is_ranging and rsi_overbought and macro_bear:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal > 0.08:
            final_signal = BASE_SIZE
        elif desired_signal < -0.08:
            final_signal = -BASE_SIZE
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