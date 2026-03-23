#!/usr/bin/env python3
"""
Experiment #1005: 1h Primary + 4h/1d HTF — Simplified Trend-Pullback with Regime Filter

Hypothesis: After 730+ failed strategies, the key insight is that 1h strategies fail due to
either (1) too many filters = 0 trades, or (2) too few filters = whipsaw losses.

This strategy uses SIMPLIFIED logic:
1. 4h HMA(21) for trend DIRECTION (only trade with HTF trend)
2. 1h RSI(14) extremes for ENTRY TIMING (pullback entries in trend direction)
3. 1d HMA(21) for macro regime filter (avoid counter-trend in strong macro)
4. ATR(14) trailing stoploss for risk management
5. NO funding rate (causes data loading issues = 0 trades)
6. NO session/volume filters (too restrictive = 0 trades)

Key difference from failed 1h strategies (#995, #998, #1000):
- REMOVED session filter (was causing 0 trades)
- REMOVED volume filter (was causing 0 trades)
- REMOVED CRSI (consistently negative Sharpe)
- SIMPLIFIED to 2 confluence factors: HTF trend + RSI extreme
- RELAXED RSI thresholds (35/65 not 30/70) to ensure trades

Why this should work:
- 4h HMA provides strong trend bias (proven in best strategy Sharpe=0.612)
- 1h RSI pullback entries = better risk/reward than breakout entries
- 1d HMA avoids major counter-trend trades
- ATR stoploss limits drawdown per trade
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: 40-70 trades/year on 1h, Sharpe > 0.612, ALL symbols positive
Timeframe: 1h (use 4h/1d for direction, 1h for entry timing)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_trend_pullback_4h1d_hma_rsi_atr_simplified_v1"
timeframe = "1h"
leverage = 1.0

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

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_50_1h = calculate_sma(close, 50)
    sma_200_1h = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === RSI SIGNALS (RELAXED thresholds for more trades) ===
        rsi_oversold = rsi_1h[i] < 35
        rsi_overbought = rsi_1h[i] > 65
        rsi_extreme_oversold = rsi_1h[i] < 25
        rsi_extreme_overbought = rsi_1h[i] > 75
        rsi_neutral = 35 <= rsi_1h[i] <= 65
        
        # === SMA FILTER (trend confirmation) ===
        above_sma50 = close[i] > sma_50_1h[i]
        below_sma50 = close[i] < sma_50_1h[i]
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 4h trend bullish + RSI pullback (most common)
        if trend_4h_bullish and rsi_oversold:
            desired_signal = BASE_SIZE
        # Secondary: 4h trend bullish + RSI extreme (stronger signal)
        elif trend_4h_bullish and rsi_extreme_oversold:
            desired_signal = BASE_SIZE
        # Tertiary: Macro bull + 4h neutral + RSI extreme (catch trend changes)
        elif macro_bull and not trend_4h_bearish and rsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        # Quaternary: Above SMA200 + RSI oversold (long-term support)
        elif above_sma200 and rsi_oversold:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 4h trend bearish + RSI rally (most common)
        if trend_4h_bearish and rsi_overbought:
            desired_signal = -BASE_SIZE
        # Secondary: 4h trend bearish + RSI extreme (stronger signal)
        elif trend_4h_bearish and rsi_extreme_overbought:
            desired_signal = -BASE_SIZE
        # Tertiary: Macro bear + 4h neutral + RSI extreme (catch trend changes)
        elif macro_bear and not trend_4h_bullish and rsi_extreme_overbought:
            desired_signal = -REDUCED_SIZE
        # Quaternary: Below SMA200 + RSI overbought (long-term resistance)
        elif below_sma200 and rsi_overbought:
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
                # Hold long if 4h trend still bullish and RSI not overbought
                if trend_4h_bullish and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses bearish + RSI overbought
            if trend_4h_bearish and rsi_1h[i] > 65:
                desired_signal = 0.0
            # Exit if macro reverses bearish + above SMA50 lost
            if macro_bear and below_sma50:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses bullish + RSI oversold
            if trend_4h_bullish and rsi_1h[i] < 35:
                desired_signal = 0.0
            # Exit if macro reverses bullish + below SMA50 lost
            if macro_bull and above_sma50:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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