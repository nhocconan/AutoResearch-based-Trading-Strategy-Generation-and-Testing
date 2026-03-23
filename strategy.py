#!/usr/bin/env python3
"""
Experiment #1010: 1h Primary + 4h/12h HTF — Regime-Adaptive CRSI + HMA Trend

Hypothesis: After 733+ failed strategies, the pattern is clear:
- Too many filters = 0 trades (experiments 998, 999, 1008)
- Too strict thresholds = no entries in bear markets
- Lower TF (1h/30m) needs VERY strict filters to avoid fee drag (>100 trades/yr = death)

This strategy uses REGIME-ADAPTIVE logic:
1. Choppiness Index (CHOP) determines regime: CHOP>55=range, CHOP<45=trend
2. 4h HMA21 provides directional bias (not filter - just bias)
3. Connors RSI for entry timing (relaxed thresholds: 20/80 not 10/90)
4. 12h HMA for macro confirmation (only one HTF trend filter)
5. ATR trailing stop at 2.5x for risk management

Key differences from failed experiments:
- SINGLE regime filter (CHOP) not multiple conflicting ones
- RELAXED CRSI thresholds to ensure trades in all market conditions
- 1h timeframe with 4h/12h HTF (proven pattern from best strategy)
- Discrete signal sizes: 0.0, ±0.20, ±0.30 (minimize fee churn)
- Target: 40-70 trades/year on 1h (use strict confluence)

Why this might work:
- CHOP regime filter adapts to market conditions (trend vs range)
- CRSI catches reversals in bear/range (2022, 2025)
- HMA trend catches bull moves (2021)
- 1h entries with 4h/12h bias = HTF trade frequency, LTF precision

Timeframe: 1h (target 40-70 trades/year)
Position Size: 0.20-0.30 (conservative for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_4h12h_hma_chop_atr_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate on mean reversion entries.
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
            streak_rsi[i] = 100 * (streak[i] / (streak[i] + 1))
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP): Measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) / (log10(n))
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR and highest high / lowest low
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh - ll > 1e-10:
            chop[i] = 100 * (tr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop_1h[i] > 55  # Range/choppy market
        regime_trend = chop_1h[i] < 45  # Trending market
        # Neutral regime: 45-55 (use both strategies)
        
        # === MACRO TREND BIAS (4h + 12h HMA21) ===
        # Both HTF must agree for strong bias
        macro_bull = (close[i] > hma_4h_aligned[i]) and (close[i] > hma_12h_aligned[i])
        macro_bear = (close[i] < hma_4h_aligned[i]) and (close[i] < hma_12h_aligned[i])
        macro_neutral = not macro_bull and not macro_bear
        
        # === CRSI SIGNALS (Connors RSI for mean reversion) ===
        # Relaxed thresholds to ensure trades
        crsi_extreme_oversold = crsi_1h[i] < 20
        crsi_extreme_overbought = crsi_1h[i] > 80
        crsi_oversold = crsi_1h[i] < 30
        crsi_overbought = crsi_1h[i] > 70
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Regime: Range (mean revert) + CRSI extreme oversold
        if regime_range and crsi_extreme_oversold:
            desired_signal = BASE_SIZE
        # Regime: Range + CRSI oversold + macro bullish
        elif regime_range and crsi_oversold and macro_bull:
            desired_signal = REDUCED_SIZE
        # Regime: Trend + CRSI oversold + macro bullish (pullback entry)
        elif regime_trend and crsi_oversold and macro_bull:
            desired_signal = BASE_SIZE
        # Regime: Neutral + CRSI extreme oversold + macro not bearish
        elif not regime_trend and crsi_extreme_oversold and not macro_bear:
            desired_signal = REDUCED_SIZE
        # Guarantee trades: CRSI extreme oversold alone (bear market entries)
        elif crsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Regime: Range (mean revert) + CRSI extreme overbought
        if regime_range and crsi_extreme_overbought:
            desired_signal = -BASE_SIZE
        # Regime: Range + CRSI overbought + macro bearish
        elif regime_range and crsi_overbought and macro_bear:
            desired_signal = -REDUCED_SIZE
        # Regime: Trend + CRSI overbought + macro bearish (pullback entry)
        elif regime_trend and crsi_overbought and macro_bear:
            desired_signal = -BASE_SIZE
        # Regime: Neutral + CRSI extreme overbought + macro not bullish
        elif not regime_trend and crsi_extreme_overbought and not macro_bull:
            desired_signal = -REDUCED_SIZE
        # Guarantee trades: CRSI extreme overbought alone (bear market entries)
        elif crsi_extreme_overbought:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not extreme overbought
                if crsi_1h[i] < 85:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not extreme oversold
                if crsi_1h[i] > 15:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI extreme overbought (take profit)
            if crsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI extreme oversold (take profit)
            if crsi_extreme_oversold:
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