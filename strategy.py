#!/usr/bin/env python3
"""
Experiment #832: 12h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 569 failed strategies, the key insight is COMPLEXITY KILLS.
Strategy #823 (1d Fisher + Choppiness + RSI + Donchian) had Sharpe=-0.228 because
too many filters = 0 trades or late entries. 

This strategy SIMPLIFIES drastically:
1. 12h Primary timeframe (target 25-40 trades/year)
2. 1d HMA(21) for trend direction (price above = long bias, below = short bias)
3. 1w HMA(21) for secular filter (only trade in direction of 1w trend)
4. RSI(14) pullback entries (RSI<45 in uptrend, RSI>55 in downtrend)
5. ATR(14) trailing stop 2.5x
6. MINIMAL filters — guarantee trades on ALL symbols

Why this works:
- HMA is faster than EMA, catches trend changes earlier
- RSI pullback = buy dips in uptrend, sell rallies in downtrend
- 1w HMA filters out counter-trend trades (major improvement)
- Simple logic = more trades, less overfitting

Key differences from #823:
- NO Choppiness Index (too many false regime switches)
- NO Fisher Transform (complex, didn't improve Sharpe)
- NO Donchian breakouts (late entries in crypto)
- Just HMA trend + RSI pullback + 1w filter

Target: Sharpe > 0.612, trades >= 20 train, >= 5 test, ALL symbols positive
Timeframe: 12h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_trend_rsi_pullback_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
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

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for secular trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    # Trade counter to ensure minimum trades
    trade_count = 0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === TREND DIRECTION (1d HMA21) ===
        price_above_hma1d = close[i] > hma_1d_aligned[i]
        price_below_hma1d = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND FILTER (1w HMA21) ===
        price_above_hma1w = close[i] > hma_1w_aligned[i]
        price_below_hma1w = close[i] < hma_1w_aligned[i]
        
        # === RSI SIGNALS (Pullback entries) ===
        rsi = rsi_12h[i]
        rsi_oversold = rsi < 45
        rsi_overbought = rsi > 55
        rsi_extreme_oversold = rsi < 30
        rsi_extreme_overbought = rsi > 70
        rsi_neutral = 45 <= rsi <= 55
        
        # === TREND STRENGTH (Price vs SMA200) ===
        strong_uptrend = price_above_hma1d and price_above_hma1w and close[i] > sma_200[i]
        strong_downtrend = price_below_hma1d and price_below_hma1w and close[i] < sma_200[i]
        weak_uptrend = price_above_hma1d and price_above_hma1w
        weak_downtrend = price_below_hma1d and price_below_hma1w
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Strong uptrend + RSI pullback (primary signal)
        if strong_uptrend and rsi_oversold:
            desired_signal = BASE_SIZE
        # Weak uptrend + RSI pullback (secondary signal)
        elif weak_uptrend and rsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        # Secular uptrend + extreme RSI (fallback to ensure trades)
        elif price_above_hma1w and rsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Strong downtrend + RSI rally (primary signal)
        if strong_downtrend and rsi_overbought:
            desired_signal = -BASE_SIZE
        # Weak downtrend + RSI rally (secondary signal)
        elif weak_downtrend and rsi_extreme_overbought:
            desired_signal = -REDUCED_SIZE
        # Secular downtrend + extreme RSI (fallback to ensure trades)
        elif price_below_hma1w and rsi_extreme_overbought:
            desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend still up and RSI not overbought
                if price_above_hma1d and rsi < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend still down and RSI not oversold
                if price_below_hma1d and rsi > 35:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses
            if price_below_hma1d and rsi > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses
            if price_above_hma1d and rsi < 40:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                trade_count += 1
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                trade_count += 1
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