#!/usr/bin/env python3
"""
Experiment #912: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 600+ failed strategies, complexity is the enemy. The current
strategy has TOO MANY conditions (CHOP regime + CRSI + RSI + Donchian + 1d HMA + 1w HMA + SMA50 + SMA200).
This creates filter fatigue where conditions rarely align, causing 0 trades or
trades only on SOL (which has strong trends).

NEW APPROACH - SIMPLER & MORE ROBUST:
1. 1d HMA(21) = PRIMARY trend filter (only one HTF, not two)
2. 12h HMA(16/48) crossover = entry trigger (proven on 12h: SOL +0.879)
3. RSI(14) filter = avoid extreme entries (30-70 range for trend continuation)
4. ATR(14) trailing stop = 2.5x for risk management
5. Discrete signals: 0.0, ±0.30 (no churn)

Why this should work better:
- Fewer filters = more trades on ALL symbols (BTC/ETH/SOL)
- HMA crossover is proven on 12h timeframe
- Single HTF (1d) avoids over-filtering
- RSI filter prevents buying tops/selling bottoms
- Target: 30-50 trades/year per symbol, Sharpe > 0.612

Critical lessons from failures:
- #902, #906, #907 all had negative Sharpe on 12h (too complex)
- Multiple HTF (1d + 1w) creates filter fatigue
- Regime switching (CHOP) adds complexity without edge
- SIMPLE trend + pullback works best on higher TF

Timeframe: 12h (target 25-40 trades/year)
Position Size: 0.30 (30% of capital, max 0.40)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crossover_rsi_1d_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator."""
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
    """Average True Range - volatility measure for stops."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[i-1]) if (j := i-1) >= 0 else high[i] - low[i])
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
    
    # Calculate primary (12h) indicators
    hma_fast_12h = calculate_hma(close, 16)
    hma_slow_12h = calculate_hma(close, 48)
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for trend bias (Rule 1 & 2)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_fast_12h[i]) or np.isnan(hma_slow_12h[i]):
            continue
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === TREND FILTER (1d HTF HMA21) ===
        # Only long if price above 1d HMA, only short if below
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === HMA CROSSOVER SIGNAL (12h) ===
        # Fast HMA crosses above slow HMA = bullish
        # Fast HMA crosses below slow HMA = bearish
        hma_bullish = hma_fast_12h[i] > hma_slow_12h[i]
        hma_bearish = hma_fast_12h[i] < hma_slow_12h[i]
        
        # Check for crossover (not just position)
        hma_cross_long = False
        hma_cross_short = False
        
        if i > 0 and not np.isnan(hma_fast_12h[i-1]) and not np.isnan(hma_slow_12h[i-1]):
            hma_cross_long = (hma_fast_12h[i-1] <= hma_slow_12h[i-1]) and (hma_fast_12h[i] > hma_slow_12h[i])
            hma_cross_short = (hma_fast_12h[i-1] >= hma_slow_12h[i-1]) and (hma_fast_12h[i] < hma_slow_12h[i])
        
        # === RSI FILTER (avoid extremes) ===
        # For long entries: RSI should be 30-70 (not overbought)
        # For short entries: RSI should be 30-70 (not oversold)
        rsi_valid_long = 30 <= rsi_12h[i] <= 70
        rsi_valid_short = 30 <= rsi_12h[i] <= 70
        
        # Allow entries even at extremes if crossover is strong
        rsi_extreme_long = rsi_12h[i] < 35  # oversold = good for long
        rsi_extreme_short = rsi_12h[i] > 65  # overbought = good for short
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: HMA cross long + trend bullish + RSI valid
        if hma_cross_long and trend_bullish and rsi_valid_long:
            desired_signal = POSITION_SIZE
        # Secondary: HMA already bullish + trend bullish + RSI extreme (pullback entry)
        elif hma_bullish and trend_bullish and rsi_extreme_long:
            desired_signal = POSITION_SIZE
        # Tertiary: HMA bullish + trend bullish (hold existing or re-enter)
        elif hma_bullish and trend_bullish and in_position and position_side > 0:
            desired_signal = POSITION_SIZE
        
        # === SHORT ENTRY ===
        # Primary: HMA cross short + trend bearish + RSI valid
        if hma_cross_short and trend_bearish and rsi_valid_short:
            desired_signal = -POSITION_SIZE
        # Secondary: HMA already bearish + trend bearish + RSI extreme (pullback entry)
        elif hma_bearish and trend_bearish and rsi_extreme_short:
            desired_signal = -POSITION_SIZE
        # Tertiary: HMA bearish + trend bearish (hold existing or re-enter)
        elif hma_bearish and trend_bearish and in_position and position_side < 0:
            desired_signal = -POSITION_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, prefer trend direction
        if trend_bullish and desired_signal < 0:
            desired_signal = 0.0
        if trend_bearish and desired_signal > 0:
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
        
        # === EXIT CONDITIONS ===
        # Exit long if trend reverses (price below 1d HMA)
        if in_position and position_side > 0 and trend_bearish:
            desired_signal = 0.0
        
        # Exit short if trend reverses (price above 1d HMA)
        if in_position and position_side < 0 and trend_bullish:
            desired_signal = 0.0
        
        # Exit if HMA crosses against position
        if in_position and position_side > 0 and hma_cross_short:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_cross_long:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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