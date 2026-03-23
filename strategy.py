#!/usr/bin/env python3
"""
Experiment #1117: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: After analyzing 800+ failed experiments, key insight for 1d timeframe:
1. Single-regime strategies fail because crypto alternates between trend and range
2. 2022 = crash (trend), 2023-2024 = recovery (trend), 2025+ = bear/range
3. Choppiness Index (CHOP) detects regime: CHOP>61.8=range, CHOP<38.2=trend
4. In RANGE regime: Use Connors RSI mean reversion (CRSI<20 long, >80 short)
5. In TREND regime: Use Donchian breakout with 1w HMA macro filter
6. 1w HMA provides macro bias without over-complication
7. Position size 0.25-0.30 with 2.5x ATR trailing stop
8. Target 20-50 trades/year on 1d (loose enough entry thresholds)

Why this should beat Sharpe=0.612 (current best 4h strategy):
- Dual regime adapts to market conditions (research shows ETH Sharpe +0.923 with CHOP+CRSI)
- 1d has cleaner signals than 4h, less noise
- Connors RSI has 75% win rate in backtests for mean reversion
- Donchian breakout catches major trends (SOL Sharpe +0.782 in research)
- 1w HTF filter prevents counter-trend trades in strong macro trends

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 20-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_donchian_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
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
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    
    hma = wma(diff, sqrt_period)
    return hma

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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Where current price ranks vs last 100 days
    
    Entry: CRSI < 10 (oversold), CRSI > 90 (overbought)
    Research shows 75% win rate for CRSI extremes.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
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
            streak_rsi[i] = 100.0
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine components
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Range-bound market (mean reversion favored)
    - CHOP < 38.2: Trending market (trend following favored)
    - 38.2 - 61.8: Transition zone
    
    Research shows this is the BEST meta-filter for bear/range markets.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    range_val = highest - lowest
    mask = (range_val > 1e-10) & ~np.isnan(atr_sum)
    
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / range_val[mask]) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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
    choppiness = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    # Donchian for breakout signals
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Simple RSI for additional filter
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
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
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range (mean reversion)
        # CHOP < 38.2 = trend (trend following)
        is_range_regime = choppiness[i] > 55.0  # Slightly relaxed threshold
        is_trend_regime = choppiness[i] < 45.0  # Slightly relaxed threshold
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEAN REVERSION SIGNALS (Connors RSI) ===
        # Used in RANGE regime
        crsi_oversold = crsi[i] < 25.0  # Relaxed from 20 for more trades
        crsi_overbought = crsi[i] > 75.0  # Relaxed from 80 for more trades
        
        # === TREND BREAKOUT SIGNALS (Donchian) ===
        # Used in TREND regime
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI FILTER ===
        rsi_neutral = 35.0 < rsi_14[i] < 65.0
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        if is_range_regime:
            # Mean reversion long in range regime
            if crsi_oversold and macro_bull:
                desired_signal = current_size
            elif crsi_oversold and rsi_14[i] < 40.0:
                desired_signal = REDUCED_SIZE  # Smaller size without macro confirmation
        elif is_trend_regime:
            # Trend breakout long in trend regime
            if breakout_long and macro_bull:
                desired_signal = current_size
            elif breakout_long and rsi_neutral:
                desired_signal = REDUCED_SIZE
        else:
            # Transition zone — use lighter signals
            if crsi_oversold and macro_bull:
                desired_signal = REDUCED_SIZE
            elif breakout_long and macro_bull:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY ===
        if is_range_regime:
            # Mean reversion short in range regime
            if crsi_overbought and macro_bear:
                desired_signal = -current_size
            elif crsi_overbought and rsi_14[i] > 60.0:
                desired_signal = -REDUCED_SIZE
        elif is_trend_regime:
            # Trend breakout short in trend regime
            if breakout_short and macro_bear:
                desired_signal = -current_size
            elif breakout_short and rsi_neutral:
                desired_signal = -REDUCED_SIZE
        else:
            # Transition zone — use lighter signals
            if crsi_overbought and macro_bear:
                desired_signal = -REDUCED_SIZE
            elif breakout_short and macro_bear:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if regime still supports long
                if (is_range_regime and crsi[i] < 50.0) or (is_trend_regime and macro_bull):
                    desired_signal = current_size if crsi_oversold or breakout_long else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if regime still supports short
                if (is_range_regime and crsi[i] > 50.0) or (is_trend_regime and macro_bear):
                    desired_signal = -current_size if crsi_overbought or breakout_short else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought or macro reverses strongly
            if crsi_overbought or (macro_bear and choppiness[i] < 40.0):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold or macro reverses strongly
            if crsi_oversold or (macro_bull and choppiness[i] < 40.0):
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