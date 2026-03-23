#!/usr/bin/env python3
"""
Experiment #801: 4h Primary + 1d HTF — Dual Regime (Mean Revert/Trend) + RSI3 + HMA

Hypothesis: After 500+ failed strategies, the key insight is:
1. Complex regime filters (CRSI, ADX, multiple conditions) cause 0 trades
2. Simple RSI(3) extremes work better than CRSI for mean reversion
3. Dual regime (chop=trend) is too complex — use SINGLE regime with adaptive logic
4. 1d HMA(21) provides stable trend bias without whipsaw
5. Relaxed entry thresholds (RSI3<20/>80, not <10/>90) generate sufficient trades
6. ATR(14) trailing stop at 2.5x protects from major drawdowns
7. Position sizing: 0.25-0.30 discrete levels

Strategy design:
1. 1d HMA(21) for trend bias (aligned via mtf_data helper)
2. 4h RSI(3) for fast mean reversion signals
3. 4h HMA(16) for short-term trend confirmation
4. 4h Bollinger Bands(20, 2.0) for overextension detection
5. 4h ATR(14) for trailing stop (2.5x)
6. Adaptive entry: mean revert against trend when overextended, follow trend on pullback
7. Discrete signals: 0.0, ±0.25, ±0.30

Key improvements from #794 failure:
- RSI(3) instead of RSI(14) — faster signals, more trades
- Simpler logic — fewer conflicting conditions
- Relaxed thresholds — RSI3<20/>80 (not <10/>90)
- Single regime with adaptive entry (not dual regime switch)
- Hold logic based on RSI3 mean reversion (exit at 50)

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi3_hma1d_adaptive_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, 16)
    rsi3_4h = calculate_rsi(close, period=3)
    rsi14_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Calculate and align 1d HMA for trend bias
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
        if np.isnan(hma_4h[i]) or np.isnan(rsi3_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(bb_sma[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === 4H HMA TREND ===
        hma_4h_bullish = close[i] > hma_4h[i]
        hma_4h_bearish = close[i] < hma_4h[i]
        
        # === RSI(3) EXTREMES (relaxed for more trades) ===
        rsi3_oversold = rsi3_4h[i] < 20
        rsi3_overbought = rsi3_4h[i] > 80
        rsi3_extreme_oversold = rsi3_4h[i] < 10
        rsi3_extreme_overbought = rsi3_4h[i] > 90
        rsi3_neutral = 40 < rsi3_4h[i] < 60
        
        # === RSI(14) FILTER ===
        rsi14_oversold = rsi14_4h[i] < 35
        rsi14_overbought = rsi14_4h[i] > 65
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        bb_width = (bb_upper[i] - bb_lower[i]) / bb_sma[i] if bb_sma[i] > 0 else 0
        bb_expanded = bb_width > 0.15  # High volatility
        
        # === ADAPTIVE ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # 1. Mean reversion: RSI3 extreme oversold + below BB (counter-trend when overextended)
        if rsi3_extreme_oversold and below_bb_lower:
            desired_signal = BASE_SIZE
        # 2. Trend pullback: 1d bullish + RSI3 oversold + above BB lower
        elif trend_1d_bullish and rsi3_oversold and not below_bb_lower:
            desired_signal = BASE_SIZE
        # 3. Moderate mean reversion: RSI3 oversold + RSI14 oversold
        elif rsi3_oversold and rsi14_oversold:
            desired_signal = REDUCED_SIZE
        # 4. Trend continuation: both HMA bullish + RSI3 neutral (pullback entry)
        elif trend_1d_bullish and hma_4h_bullish and rsi3_neutral:
            desired_signal = REDUCED_SIZE
        
        # SHORT ENTRIES
        # 1. Mean reversion: RSI3 extreme overbought + above BB (counter-trend when overextended)
        if rsi3_extreme_overbought and above_bb_upper:
            desired_signal = -BASE_SIZE
        # 2. Trend pullback: 1d bearish + RSI3 overbought + below BB upper
        elif trend_1d_bearish and rsi3_overbought and not above_bb_upper:
            desired_signal = -BASE_SIZE
        # 3. Moderate mean reversion: RSI3 overbought + RSI14 overbought
        elif rsi3_overbought and rsi14_overbought:
            desired_signal = -REDUCED_SIZE
        # 4. Trend continuation: both HMA bearish + RSI3 neutral (pullback entry)
        elif trend_1d_bearish and hma_4h_bearish and rsi3_neutral:
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
        
        # === EXIT CONDITIONS (RSI3 mean reversion exit) ===
        if in_position and position_side > 0:
            # Exit long when RSI3 mean-reverts to neutral or overbought
            if rsi3_4h[i] > 60:
                desired_signal = 0.0
            # Exit if 1d trend reverses strongly
            if trend_1d_bearish and rsi3_4h[i] > 50:
                desired_signal = 0.0
            # Exit if price hits BB upper in high vol
            if bb_expanded and above_bb_upper:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short when RSI3 mean-reverts to neutral or oversold
            if rsi3_4h[i] < 40:
                desired_signal = 0.0
            # Exit if 1d trend reverses strongly
            if trend_1d_bullish and rsi3_4h[i] < 50:
                desired_signal = 0.0
            # Exit if price hits BB lower in high vol
            if bb_expanded and below_bb_lower:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if RSI3 not overbought and trend intact
                if rsi3_4h[i] < 65 and (trend_1d_bullish or hma_4h_bullish):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if RSI3 not oversold and trend intact
                if rsi3_4h[i] > 35 and (trend_1d_bearish or hma_4h_bearish):
                    desired_signal = -BASE_SIZE
        
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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