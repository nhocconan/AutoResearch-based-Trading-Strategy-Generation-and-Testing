#!/usr/bin/env python3
"""
Experiment #388: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Recent failures (#376-#387) show 0 trades due to overly complex entry
conditions. This version SIMPLIES dramatically:
1. HTF (12h/1d) for trend BIAS only (not entry filter)
2. 4h HMA for primary trend direction
3. RSI pullback (40-60 range) for entry timing
4. ATR stoploss at 2.5x
5. MAX 3 confluence conditions per entry

Key difference from failed strategies:
- NO Choppiness Index (adds complexity, rarely triggers)
- NO Connors RSI (failed in #378, #380, #382, #384)
- NO ADX regime switching (failed in #387)
- Just HMA trend + RSI pullback + HTF bias

Entry Logic (SIMPLE):
- Long: 4h HMA bull + 12h HMA bull + RSI 35-55 + price near HMA
- Short: 4h HMA bear + 12h HMA bear + RSI 45-65 + price near HMA
- Exit: HMA flip OR 2.5x ATR stoploss

Target: Sharpe>0.5, DD>-30%, trades>=30 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h1d_v1"
timeframe = "4h"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (12h + 1d) ===
        # Both must agree for strong signal
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias: both 12h and 1d agree
        htf_strong_bull = htf_12h_bull and htf_1d_bull
        htf_strong_bear = htf_12h_bear and htf_1d_bear
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === HMA CROSSOVER (entry trigger) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_4h_fast[i]) and not np.isnan(hma_4h_fast[i-1]):
            if not np.isnan(hma_4h[i]) and not np.isnan(hma_4h[i-1]):
                if hma_4h_fast[i-1] <= hma_4h[i-1] and hma_4h_fast[i] > hma_4h[i]:
                    hma_cross_long = True
                if hma_4h_fast[i-1] >= hma_4h[i-1] and hma_4h_fast[i] < hma_4h[i]:
                    hma_cross_short = True
        
        # === RSI PULLBACK ZONE (entry timing) ===
        # Long: RSI pulled back to 35-55 (not oversold, just cooling off)
        # Short: RSI pulled back to 45-65 (not overbought, just cooling off)
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        
        # === PRICE NEAR HMA (pullback confirmation) ===
        # Price within 2% of HMA = pullback, not breakout
        hma_distance = abs(close[i] - hma_4h[i]) / hma_4h[i] if hma_4h[i] > 1e-10 else 999
        price_near_hma = hma_distance < 0.02
        
        # === DONCHIAN BREAKOUT (alternative entry) ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === ENTRY LOGIC (SIMPLIFIED - 2-3 conditions max) ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bull + 4h bull + (RSI pullback OR HMA cross)
        if htf_strong_bull and hma_bull:
            if rsi_pullback_long and price_near_hma:
                desired_signal = SIZE_STRONG
            elif hma_cross_long:
                desired_signal = SIZE_BASE
            elif breakout_long and above_sma200:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: HTF bear + 4h bear + (RSI pullback OR HMA cross)
        elif htf_strong_bear and hma_bear:
            if rsi_pullback_short and price_near_hma:
                desired_signal = -SIZE_STRONG
            elif hma_cross_short:
                desired_signal = -SIZE_BASE
            elif breakout_short and below_sma200:
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