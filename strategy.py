#!/usr/bin/env python3
"""
Experiment #097: 1d Primary + 1w HTF — Dual Regime with Loose Filters

Hypothesis: After 82 failed experiments, the pattern is clear:
- Complex filter combinations = 0 trades (#087, #088, #089, #090, #093, #096 all Sharpe=0.000)
- Simple HMA crossover works but needs regime filter to avoid chop losses
- 1d timeframe is underexplored — should generate 20-50 trades/year naturally
- Dual regime (mean revert in chop, trend follow otherwise) proven on ETH (Sharpe +0.923)

This strategy uses LOOSE dual regime on 1d:
1. 1w HMA = major trend bias (only 1 HTF filter)
2. Choppiness Index (14) = regime detection (loose: >55 = chop, <45 = trend)
3. CHOP regime: Connors RSI mean reversion (CRSI<15 long, >85 short)
4. TREND regime: HMA(16/48) crossover with RSI filter (>40 long, <60 short)
5. ATR trailing stoploss (2.5x) for risk management
6. Position size: 0.28 (28% of capital)

Key design choices:
- Timeframe: 1d (natural 20-50 trades/year, minimal fee drag)
- HTF: 1w for major trend bias (slower, more stable than 1d)
- Choppiness thresholds: 55/45 (not 61.8/38.2, ensures more regime switches)
- CRSI thresholds: 15/85 (not 10/90, ensures more mean reversion trades)
- RSI thresholds: 40/60 (not 30/70, ensures more trend trades)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = close_series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            chop[i] = 100.0
        else:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    Used for mean reversion entries
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values (treat as price series)
    streak_rsi = calculate_rsi(np.abs(streak) + 1, streak_period)  # +1 to avoid zero
    
    # PercentRank: percentage of closes in last 100 days that are below current close
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)  # exclude current from comparison
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # CRSI = average of 3 components
    crsi = (rsi_close + streak_rsi + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 1d)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # LOOSE thresholds: >55 = chop, <45 = trend (not 61.8/38.2)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        # Neutral zone (45-55): use trend logic as default
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME: Use Connors RSI
            # Long: CRSI < 15 (oversold) + HTF not strongly bearish
            # Short: CRSI > 85 (overbought) + HTF not strongly bullish
            if crsi[i] < 15.0 and not htf_bear:
                desired_signal = SIZE
            elif crsi[i] > 85.0 and not htf_bull:
                desired_signal = -SIZE
        else:
            # TREND REGIME (or neutral): Use HMA crossover + RSI filter
            hma_cross_bull = hma_fast[i] > hma_slow[i]
            hma_cross_bear = hma_fast[i] < hma_slow[i]
            
            # LOOSE RSI filters: >40 for long, <60 for short
            rsi_ok_long = rsi[i] > 40.0
            rsi_ok_short = rsi[i] < 60.0
            
            # LONG: HMA bull cross + RSI > 40 + HTF bull (or neutral)
            if hma_cross_bull and rsi_ok_long and (htf_bull or not is_trending):
                desired_signal = SIZE
            # SHORT: HMA bear cross + RSI < 60 + HTF bear (or neutral)
            elif hma_cross_bear and rsi_ok_short and (htf_bear or not is_trending):
                desired_signal = -SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals