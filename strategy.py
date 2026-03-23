#!/usr/bin/env python3
"""
Experiment #1194: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Current #1184 (Sharpe=-0.388) fails due to overly complex regime-switching logic
with Choppiness + CRSI + Bollinger creating conflicting signals. This version simplifies to
proven pattern: HTF trend direction + LTF pullback entries.

Key changes from #1184:
1. REMOVE Choppiness Index regime switching (creates false chop/trend signals)
2. REMOVE CRSI complexity (use standard RSI(14) instead — more reliable)
3. REMOVE Bollinger Bands (redundant with RSI for mean reversion)
4. SIMPLER logic: 1d HMA(50) for macro trend + 4h HMA(21) for primary trend + RSI pullback
5. LOOSEN entry conditions to ensure 30-50 trades/year (RSI 35-65 range, not extremes)

Entry Logic:
- Long: 1d HMA bullish + 4h HMA bullish + RSI(14) pulls back to 40-50 zone
- Short: 1d HMA bearish + 4h HMA bearish + RSI(14) rallies to 50-60 zone
- Exit: RSI crosses back through 50 OR stoploss hit (2.5x ATR)

Target: 30-50 trades/year, Sharpe > 0.612 (beat current best)
Position Size: 0.30 discrete
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simplified_hma_rsi_pullback_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average for long-term trend filter."""
    n = len(close)
    sma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(hma_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA50) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA21) ===
        trend_bull = close[i] > hma_4h[i]
        trend_bear = close[i] < hma_4h[i]
        
        # === HMA CONFIRMATION (fast vs slow) ===
        hma_bull = hma_4h_fast[i] > hma_4h[i] if not np.isnan(hma_4h_fast[i]) else False
        hma_bear = hma_4h_fast[i] < hma_4h[i] if not np.isnan(hma_4h_fast[i]) else False
        
        # === LONG-TERM FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK ZONES ===
        # Long: RSI pulled back to 40-50 zone in uptrend
        rsi_long_pullback = 38.0 < rsi[i] < 52.0
        # Short: RSI rallied to 50-62 zone in downtrend
        rsi_short_pullback = 48.0 < rsi[i] < 62.0
        
        # === RSI EXIT ZONES ===
        rsi_long_exit = rsi[i] > 58.0
        rsi_short_exit = rsi[i] < 42.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # LONG ENTRY: All trend filters aligned + RSI pullback
        if macro_bull and trend_bull and above_sma200 and rsi_long_pullback:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRY: All trend filters aligned + RSI pullback
        elif macro_bear and trend_bear and below_sma200 and rsi_short_pullback:
            desired_signal = -BASE_SIZE
        
        # === EXISTING POSITION MANAGEMENT ===
        if in_position:
            if position_side > 0:
                # Long position: exit if RSI overbought or trend breaks
                if rsi_long_exit or close[i] < hma_4h[i]:
                    desired_signal = 0.0
            elif position_side < 0:
                # Short position: exit if RSI oversold or trend breaks
                if rsi_short_exit or close[i] > hma_4h[i]:
                    desired_signal = 0.0
        
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
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals