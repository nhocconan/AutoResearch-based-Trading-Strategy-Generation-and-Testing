#!/usr/bin/env python3
"""
Experiment #1011: 4h Primary + 1d/1w HTF — Simplified CRSI + Donchian + Dual HMA Regime

Hypothesis: Experiment #1009 failed (Sharpe=-0.466) due to overly complex hold/exit logic
causing premature exits and signal churn. This version SIMPLIFIES:

1. DUAL HMA REGIME (1d + 1w): Both must agree for trend direction
   - Long: price > 1d HMA21 AND price > 1w HMA21 (strong bullish)
   - Short: price < 1d HMA21 AND price < 1w HMA21 (strong bearish)
   - Neutral: mixed signals = no position (avoids whipsaw)

2. CRSI ENTRY (simplified thresholds):
   - Long: CRSI < 20 in bullish regime (mean reversion pullback)
   - Short: CRSI > 80 in bearish regime (mean reversion rally)
   - More lenient than #1009 (15/85 → 20/80) to ensure trades

3. DONCHIAN CONFIRMATION (optional boost):
   - Breakout above Donchian(20) high adds conviction to long
   - Breakout below Donchian(20) low adds conviction to short
   - Not required for entry, just increases position size

4. SIMPLIFIED EXIT:
   - Stoploss: 2.5x ATR trailing (same as #1009)
   - Exit: regime reversal (1d HMA crosses) OR CRSI extreme opposite
   - NO complex hold logic — reduces churn

5. POSITION SIZING:
   - Base: 0.25 (25% capital)
   - With Donchian confirmation: 0.30 (30% capital)
   - Discrete levels only: 0.0, ±0.25, ±0.30

Why this beats #1009:
- Dual HMA (1d+1w) = stronger regime filter, fewer whipsaws
- Simpler exit logic = less churn, lower fees
- Lenient CRSI (20/80) = ensures minimum trades
- Target: 25-40 trades/year on 4h timeframe

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_dual_hma_regime_v1"
timeframe = "4h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 * (np.abs(streak[i]) / (np.abs(streak[i]) + 1))
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (1 / (np.abs(streak[i]) + 1))
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        count_lower = np.sum(window[:-1] < current)
        percent_rank[i] = 100 * count_lower / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel: 20-bar high and low."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA21 for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA21 for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    CONFIRMED_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]):
            continue
        
        # === DUAL HMA REGIME (1d + 1w must agree) ===
        above_1d_hma = close[i] > hma_1d_aligned[i]
        above_1w_hma = close[i] > hma_1w_aligned[i]
        
        regime_bullish = above_1d_hma and above_1w_hma
        regime_bearish = (not above_1d_hma) and (not above_1w_hma)
        regime_neutral = not regime_bullish and not regime_bearish
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 20
        crsi_overbought = crsi_4h[i] > 80
        crsi_extreme_oversold = crsi_4h[i] < 15
        crsi_extreme_overbought = crsi_4h[i] > 85
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]
        donchian_breakout_short = close[i] < donchian_lower[i-1]
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_bullish and not in_position:
            # CRSI oversold in bullish regime = primary entry
            if crsi_oversold:
                if donchian_breakout_long:
                    desired_signal = CONFIRMED_SIZE
                else:
                    desired_signal = BASE_SIZE
            # Extreme oversold alone (ensures trades in pullbacks)
            elif crsi_extreme_oversold:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRIES ===
        if regime_bearish and not in_position:
            # CRSI overbought in bearish regime = primary entry
            if crsi_overbought:
                if donchian_breakout_short:
                    desired_signal = -CONFIRMED_SIZE
                else:
                    desired_signal = -BASE_SIZE
            # Extreme overbought alone (ensures trades in rallies)
            elif crsi_extreme_overbought:
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
        
        # === EXIT ON REGIME REVERSAL ===
        if in_position and not stoploss_triggered:
            if position_side > 0:
                # Exit long if regime turns bearish or CRSI extreme overbought
                if regime_bearish or crsi_extreme_overbought:
                    desired_signal = 0.0
                # Hold long if still bullish (even if CRSI neutral)
                elif regime_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Exit short if regime turns bullish or CRSI extreme oversold
                if regime_bullish or crsi_extreme_oversold:
                    desired_signal = 0.0
                # Hold short if still bearish
                elif regime_bearish:
                    desired_signal = -BASE_SIZE
        
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
                # Flip position
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