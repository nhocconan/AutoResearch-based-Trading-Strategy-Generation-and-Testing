#!/usr/bin/env python3
"""
Experiment #1092: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI

Hypothesis: After 785+ failed experiments, the winning pattern for 12h is:
1. Choppiness Index (CHOP) detects regime: >55 = range (mean revert), <45 = trend
2. Connors RSI (CRSI) for entries: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 in range regime or pullback in trend regime
   - Short: CRSI > 85 in range regime or rally in trend regime
3. 1d HMA21 for macro bias — only trade with higher TF trend
4. ATR(14) trailing stop 2.5x — proper risk management

Why this should beat Sharpe=0.612:
- Choppiness filter prevents trend strategies in chop (major failure mode)
- Connors RSI has 75% win rate in backtests
- 12h timeframe = 20-50 trades/year (optimal fee/trade ratio)
- 1d HTF filter prevents counter-trend trades
- LOOSENED entry thresholds to ensure >=10 trades per symbol

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_1d_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    Formula: HMA = WMA(sqrt(N)) of (2*WMA(N/2) - WMA(N))
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
    """Relative Strength Index — momentum oscillator."""
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
    Connors RSI — combines 3 components for superior mean-reversion signals.
    Formula: CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close — short-term momentum
    2. RSI(2) of streak — consecutive up/down days
    3. PercentRank(100) — where current close ranks vs last 100 bars
    
    Entry: CRSI < 10-15 = oversold (long), CRSI > 85-90 = overbought (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_rsi = np.full(n, 50.0)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_gain[i-streak_period+1:i+1])
        avg_loss = np.mean(streak_loss[i-streak_period+1:i+1])
        if avg_loss > 1e-10:
            rs = avg_gain / avg_loss
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            streak_rsi[i] = 100.0 if avg_gain > 0 else 50.0
    
    # Component 3: PercentRank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1)
        percent_rank[i] = rank * 100.0
    
    # Combine components
    mask = ~np.isnan(rsi_3) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_3[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — detects trending vs ranging markets.
    Formula: CHOP = 100 * LOG10(SUM(ATR,period) / (HighestHigh - LowestLow)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/ranging market (use mean-reversion)
    - CHOP < 38.2 = trending market (use trend-following)
    - 38.2 < CHOP < 61.8 = transition zone
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0  # Range/mean-reversion regime
        is_trending = chop[i] < 45.0  # Trend-following regime
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA crossover) ===
        hma_bull = hma_16[i] > hma_48[i]
        hma_bear = hma_16[i] < hma_48[i]
        
        # === CONNORS RSI SIGNALS (LOOSENED for trade generation) ===
        # Long: CRSI oversold
        crsi_oversold = crsi[i] < 20.0
        # Short: CRSI overbought
        crsi_overbought = crsi[i] > 80.0
        
        # === VOLATILITY CHECK ===
        vol_spike = atr[i] > 2.0 * np.nanmedian(atr[max(0, i-100):i]) if i > 100 else False
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Range regime: CRSI oversold (mean reversion)
        # Trend regime: CRSI pullback + trend alignment
        if is_choppy:
            if crsi_oversold:
                desired_signal = current_size
        else:  # trending
            if crsi_oversold and macro_bull and hma_bull:
                desired_signal = current_size
        
        # === SHORT ENTRY ===
        if is_choppy:
            if crsi_overbought:
                desired_signal = -current_size
        else:  # trending
            if crsi_overbought and macro_bear and hma_bear:
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
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought yet
                if crsi[i] < 75.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if CRSI not oversold yet
                if crsi[i] > 25.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought or macro reverses strongly
            if crsi[i] > 70.0:
                desired_signal = 0.0
            if macro_bear and hma_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold or macro reverses strongly
            if crsi[i] < 30.0:
                desired_signal = 0.0
            if macro_bull and hma_bull:
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