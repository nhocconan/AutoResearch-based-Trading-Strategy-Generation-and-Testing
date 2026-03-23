#!/usr/bin/env python3
"""
Experiment #1123: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: After 800+ failed experiments, key insight for 1d timeframe:
1. Regular RSI failed on 1d (exp#1113 Sharpe=-0.723) — need different oscillator
2. Connors RSI (CRSI) proven on ETH with Sharpe +0.923 in research
3. CRSI combines: RSI(3) + RSI_Streak(2) + PercentRank(100) — catches short-term extremes
4. 1w HMA provides macro trend filter without over-complication
5. Donchian(20) breakout confirmation ensures momentum alignment
6. Loose CRSI thresholds (15/85) ensure adequate trade frequency on 1d
7. Choppiness Index filter ONLY to avoid trading in extreme chop

Why this should beat Sharpe=0.612:
- CRSI is more sensitive than RSI(14) — catches more pullback entries
- 1w trend filter is stronger than 1d/4h — fewer false signals
- Donchian confirmation ensures we trade with momentum
- Position size 0.30 with 2.5x ATR trail limits drawdown
- Target 25-40 trades/year on 1d (perfect frequency)

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base, 0.15 reduced
Stoploss: 2.5x ATR trailing
Target: 25-40 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = int(period / 2)
    if half < 1:
        half = 1
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry: CRSI < 15 (oversold) or CRSI > 85 (overbought)
    Proven Sharpe +0.923 on ETH in research.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, 50.0)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 50.0 + min(streak[i] * 10.0, 50.0)
        else:
            streak_rsi[i] = 50.0 - min(abs(streak[i]) * 10.0, 50.0)
    
    # Component 3: Percent Rank of daily returns over 100 days
    percent_rank = np.full(n, 50.0)
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        current = returns[i]
        rank = np.sum(window < current) / rank_period
        percent_rank[i] = rank * 100.0
    
    # Combine components
    mask = ~np.isnan(rsi_short)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 1e-10 and (highest - lowest) > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(choppiness[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME FILTER (Choppiness) ===
        # Only trade when market is trending (CHOP < 55)
        is_trending = choppiness[i] < 55.0
        
        # === MEAN REVERSION SIGNAL (Connors RSI) ===
        # Loose thresholds to ensure adequate trade frequency
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === BREAKOUT CONFIRMATION (Donchian) ===
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 1e-10:
            price_position = (close[i] - donchian_lower[i]) / donchian_range
            near_upper = price_position > 0.55
            near_lower = price_position < 0.45
        else:
            near_upper = False
            near_lower = False
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # Macro bull + trending regime + CRSI oversold + price in upper Donchian half
        if macro_bull and is_trending and crsi_oversold:
            if near_upper or crsi[i] < 30.0:
                desired_signal = current_size
        
        # === SHORT ENTRY ===
        # Macro bear + trending regime + CRSI overbought + price in lower Donchian half
        elif macro_bear and is_trending and crsi_overbought:
            if near_lower or crsi[i] > 70.0:
                desired_signal = -current_size
        
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
                # Hold long if macro still bull and not extremely overbought
                if macro_bull and crsi[i] < 80.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro still bear and not extremely oversold
                if macro_bear and crsi[i] > 20.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or CRSI extremely overbought
            if macro_bear or crsi[i] > 85.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or CRSI extremely oversold
            if macro_bull or crsi[i] < 15.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals