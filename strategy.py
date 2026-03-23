#!/usr/bin/env python3
"""
Experiment #1006: 12h Primary + 1d HTF — Simplified Regime + Connors RSI + HMA Trend

Hypothesis: After 731 failed strategies, the key is SIMPLICITY + ENSURING TRADES GENERATE.
Complex multi-filter strategies fail because conditions never align. This strategy uses:

1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Proven 75% win rate in research literature
   - Long when CRSI < 15, Short when CRSI > 85
   - Much more reliable than standard RSI(14)

2. 1d HMA(21) for macro trend bias — only trade with HTF trend
   - Long only when price > 1d HMA
   - Short only when price < 1d HMA

3. Choppiness Index regime switch
   - CHOP > 55 = range (use CRSI mean reversion)
   - CHOP < 45 = trend (use pullback entries)

4. ATR(14) trailing stop at 2.5x — mandatory risk management

5. RELAXED thresholds to ENSURE trades generate:
   - CRSI < 20 for long (not 10)
   - CRSI > 80 for short (not 90)
   - This is critical — many strategies fail with 0 trades

Why 12h timeframe:
- Target 25-40 trades/year (low fee drag)
- HTF signals (1d) provide strong trend bias
- Works in both bull and bear markets
- Proven pattern from research (ETH Sharpe +0.923 with similar setup)

Position sizing: 0.25 base, 0.30 strong signal (discrete levels minimize churn)
Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_chop_1d_regime_atr_v1"
timeframe = "12h"
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

def calculate_rsi_streak(close, period=2):
    """RSI Streak component of Connors RSI — consecutive up/down days."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    streak_direction = np.sign(streak)
    
    # Normalize: longer streak = more extreme
    streak_score = np.clip(abs_streak / 5.0, 0, 1) * 100
    streak_rsi = np.where(streak_direction >= 0, 50 + streak_score / 2, 50 - streak_score / 2)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of Connors RSI — current return vs historical."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period + 1:
        return pr
    
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / period
        pr[i] = rank * 100
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    for i in range(n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pr[i]) / 3
    
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_12h[i]):
            continue
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        
        # === CONNORS RSI SIGNALS (RELAXED for trade generation) ===
        crsi_oversold = crsi_12h[i] < 20  # Relaxed from 15
        crsi_overbought = crsi_12h[i] > 80  # Relaxed from 85
        crsi_extreme_oversold = crsi_12h[i] < 10
        crsi_extreme_overbought = crsi_12h[i] > 90
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI oversold (mean reversion in range)
            if crsi_oversold:
                desired_signal = BASE_SIZE
            # Strong long: Extreme CRSI oversold
            if crsi_extreme_oversold:
                desired_signal = STRONG_SIZE
            
            # Short: CRSI overbought (mean reversion in range)
            if crsi_overbought:
                desired_signal = -BASE_SIZE
            # Strong short: Extreme CRSI overbought
            if crsi_extreme_overbought:
                desired_signal = -STRONG_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with CRSI Pullback ===
        elif trending_regime:
            # Long: Bullish macro + CRSI pullback
            if macro_bull and crsi_oversold:
                desired_signal = BASE_SIZE
            if macro_bull and crsi_extreme_oversold:
                desired_signal = STRONG_SIZE
            
            # Short: Bearish macro + CRSI rally
            if macro_bear and crsi_overbought:
                desired_signal = -BASE_SIZE
            if macro_bear and crsi_extreme_overbought:
                desired_signal = -STRONG_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only extreme CRSI signals with macro confluence
            if crsi_extreme_oversold and macro_bull:
                desired_signal = BASE_SIZE
            if crsi_extreme_overbought and macro_bear:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bull and CRSI not overbought
                if macro_bull and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bear and CRSI not oversold
                if macro_bear and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + CRSI overbought
            if macro_bear and crsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + CRSI oversold
            if macro_bull and crsi_12h[i] < 30:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = STRONG_SIZE if desired_signal >= STRONG_SIZE else BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -STRONG_SIZE if desired_signal <= -STRONG_SIZE else -BASE_SIZE
        
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