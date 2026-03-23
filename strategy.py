#!/usr/bin/env python3
"""
Experiment #783: 1d Primary + 1w HTF — Simple Trend + RSI Pullback + Donchian Breakout

Hypothesis: After 531 failed strategies and analyzing the failure patterns:
1. 1d timeframe naturally filters noise — fewer but higher quality signals
2. Weekly HMA(21) provides strong, reliable trend bias (slow but accurate)
3. Daily RSI(14) pullbacks into trend direction work better than complex CRSI
4. Donchian(20) breakout confirms momentum — avoids fakeouts
5. Simpler logic = more trades (critical: need ≥10 train, ≥3 test per symbol)
6. ATR(14) trailing stop at 2.5x protects from 2022-style crashes
7. Position size 0.25-0.30 discrete levels minimize fee churn

Why this differs from failed attempts:
- #771-#782 all used 4h/12h/30m — 1d is underexplored and more stable
- Removed complex regime switching (Choppiness, ADX hysteresis) — caused 0 trades
- Removed volume filters — too restrictive for daily bars
- Simpler RSI thresholds (40/60 vs 15/85) — generates more signals
- Weekly trend filter is stronger than 12h/4h for daily entries

Strategy design:
1. 1w HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 1d RSI(14) for entry timing (long <40, short >60)
3. 1d Donchian(20) breakout confirmation (price > 20d high for long)
4. 1d ATR(14) for trailing stop (2.5x from entry)
5. Discrete signals: 0.0, ±0.25, ±0.30
6. Position sizing: 0.25 base, 0.30 with strong confirmation

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 10-30 trades/year per symbol)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster response than EMA with less lag.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(data, w_period):
        weights = np.arange(1, w_period + 1)
        weights = weights / weights.sum()
        result = np.full(len(data), np.nan)
        for i in range(w_period - 1, len(data)):
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    half = period // 2
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    
    # Combine
    combined = 2 * wma_half - wma_full
    
    # Final WMA with sqrt period
    sqrt_period = int(np.sqrt(period))
    hma = wma(combined, sqrt_period)
    
    return hma

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

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === SECONDARY TREND FILTER (1d SMA50) ===
        trend_1d_bullish = close[i] > sma_50[i]
        trend_1d_bearish = close[i] < sma_50[i]
        
        # === LONG-TERM FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_1d[i] < 40
        rsi_overbought = rsi_1d[i] > 60
        rsi_extreme_oversold = rsi_1d[i] < 30
        rsi_extreme_overbought = rsi_1d[i] > 70
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: Weekly bullish + RSI pullback + Donchian confirmation
        if trend_1w_bullish and rsi_oversold:
            if donchian_breakout_long:
                desired_signal = STRONG_SIZE
            elif trend_1d_bullish:
                desired_signal = BASE_SIZE
        
        # Secondary: Weekly bullish + extreme RSI + above SMA200
        if trend_1w_bullish and rsi_extreme_oversold and above_sma200:
            if desired_signal < BASE_SIZE:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: Weekly bearish + RSI rally + Donchian confirmation
        if trend_1w_bearish and rsi_overbought:
            if donchian_breakout_short:
                desired_signal = -STRONG_SIZE
            elif trend_1d_bearish:
                desired_signal = -BASE_SIZE
        
        # Secondary: Weekly bearish + extreme RSI + below SMA200
        if trend_1w_bearish and rsi_extreme_overbought and below_sma200:
            if desired_signal > -BASE_SIZE:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly trend intact and RSI not overbought
                if trend_1w_bullish and rsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly trend intact and RSI not oversold
                if trend_1w_bearish and rsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses or RSI overbought
            if trend_1w_bearish and rsi_1d[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses or RSI oversold
            if trend_1w_bullish and rsi_1d[i] < 35:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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