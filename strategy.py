#!/usr/bin/env python3
"""
Experiment #744: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Dual-Regime

Hypothesis: Previous 12h experiments failed due to either (a) too many regime filters
blocking all trades, or (b) overly loose entries with no edge. This strategy uses:

1. CONNORS RSI (CRSI) for entries - proven 75% win rate in mean reversion
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 15, Short: CRSI > 85

2. CHOPPINESS INDEX for regime detection
   CHOP > 61.8 = range (use CRSI mean reversion)
   CHOP < 38.2 = trend (use HMA trend following)
   Between = neutral (reduce size or flat)

3. 1d HMA(21) for HTF bias - only trade in direction of daily trend
4. 1w HMA(21) for ultra-HTF filter - avoid counter-trend in weekly

5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30 (max 0.35)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines short-term RSI, streak strength, and percentile rank
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period + rsi_period:
        return np.full(n, np.nan)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100.0 * streak[i] / max(streak_period, streak[i])
        else:
            streak_rsi[i] = 100.0 * (streak_period + streak[i]) / streak_period
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percentile Rank - where current close ranks in last N periods
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window < close[i])
        pct_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy (mean reversion favored)
    CHOP < 38.2 = trending (trend following favored)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    crsi_14 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    SIZE_MAX = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(crsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        is_choppy = chop_value > 61.8  # Range/mean reversion regime
        is_trending = chop_value < 38.2  # Trend following regime
        is_neutral = not is_choppy and not is_trending
        
        # === 12h HMA TREND ===
        hma_12h_bull = hma_16[i] > hma_48[i]
        hma_12h_bear = hma_16[i] < hma_48[i]
        
        # === HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === CRSI EXTREMES (Mean Reversion Signals) ===
        crsi_oversold = crsi_14[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi_14[i] > 85.0  # Extreme overbought
        crsi_mild_oversold = crsi_14[i] < 25.0
        crsi_mild_overbought = crsi_14[i] > 75.0
        
        # === DUAL REGIME ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if htf_1d_bull and htf_1w_bull:  # Both HTF bullish
            if is_choppy:  # Mean reversion regime
                if crsi_oversold:
                    desired_signal = SIZE_STRONG
                elif crsi_mild_oversold and hma_12h_bull:
                    desired_signal = SIZE_BASE
            elif is_trending:  # Trend following regime
                if hma_crossover_long:
                    desired_signal = SIZE_STRONG
                elif hma_12h_bull and crsi_mild_oversold:
                    desired_signal = SIZE_BASE
            else:  # Neutral regime - reduced size
                if crsi_oversold and hma_12h_bull:
                    desired_signal = SIZE_BASE * 0.7
        
        # SHORT ENTRIES
        elif htf_1d_bear and htf_1w_bear:  # Both HTF bearish
            if is_choppy:  # Mean reversion regime
                if crsi_overbought:
                    desired_signal = -SIZE_STRONG
                elif crsi_mild_overbought and hma_12h_bear:
                    desired_signal = -SIZE_BASE
            elif is_trending:  # Trend following regime
                if hma_crossover_short:
                    desired_signal = -SIZE_STRONG
                elif hma_12h_bear and crsi_mild_overbought:
                    desired_signal = -SIZE_BASE
            else:  # Neutral regime - reduced size
                if crsi_overbought and hma_12h_bear:
                    desired_signal = -SIZE_BASE * 0.7
        
        # Mixed HTF signals - only take strong mean reversion
        elif htf_1d_bull and htf_1w_bear:
            if is_choppy and crsi_oversold:
                desired_signal = SIZE_BASE * 0.5
        elif htf_1d_bear and htf_1w_bull:
            if is_choppy and crsi_overbought:
                desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = min(SIZE_MAX, SIZE_STRONG)
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = max(-SIZE_MAX, -SIZE_STRONG)
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals