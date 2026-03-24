#!/usr/bin/env python3
"""
Experiment #077: 1d Primary + 1w HTF — Choppiness Index Regime + Connors RSI

Hypothesis: After 76 failed experiments, the key insight from research is that
Choppiness Index + Connors RSI achieved ETH Sharpe +0.923 in backtests.
This strategy uses:
1. 1d primary timeframe (low fee drag, 20-50 trades/year target)
2. 1w HMA for major trend bias (very loose filter)
3. Choppiness Index(14) regime detection: >61.8 = range, <38.2 = trend
4. Connors RSI for entries: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
5. LOOSE thresholds to ensure trades generate (learned from 0-trade failures)
6. Discrete sizing: 0.30 with 2.5x ATR trailing stop

Why this should work:
- 1d timeframe = minimal fee drag, proven higher TF works best
- Choppiness Index switches between mean-revert and trend-follow modes
- Connors RSI has 75% win rate in research for catching reversals
- 1w HMA is very loose filter (won't block trades like strict 4h HMA)
- Simple logic = fewer bugs, more reliable execution

Entry Logic:
- Range regime (CHOP>61.8): Long when CRSI<20, Short when CRSI>80
- Trend regime (CHOP<38.2): Long when CRSI<40 + price>1w HMA, Short when CRSI>60 + price<1w HMA
- Size: 0.30 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - identifies trending vs ranging markets
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
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
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_crsi(close):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10-20, Short when CRSI > 80-90
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    rsi_3 = np.full(n, np.nan)
    for i in range(3, n):
        if avg_loss[i] < 1e-10:
            rsi_3[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_3[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak (use absolute values for up streaks, negative for down)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    for i in range(2, n):
        total_avg = avg_streak_gain[i] + avg_streak_loss[i]
        if total_avg < 1e-10:
            streak_rsi[i] = 50.0
        elif avg_streak_loss[i] < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: Percent Rank of daily returns over 100 days
    percent_rank = np.full(n, np.nan)
    daily_returns = np.diff(close) / close[:-1]
    daily_returns = np.concatenate([[0.0], daily_returns])
    
    for i in range(100, n):
        window = daily_returns[i - 99:i + 1]  # 100-day window
        current_return = daily_returns[i]
        rank = np.sum(window < current_return) / len(window)
        percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias (very loose filter)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 150 bars for CRSI (100) + CHOP (14) + warmup
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 61.8  # Mean revert regime
        is_trend = chop[i] < 38.2  # Trend follow regime
        # Between 38.2-61.8 = neutral, use looser criteria
        
        # === HTF TREND BIAS (1w HMA - very loose) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # RANGE REGIME (mean revert) - LOOSE thresholds
        if is_range:
            # Long when CRSI extremely low (oversold in range)
            if crsi[i] < 25.0:
                desired_signal = SIZE
            # Short when CRSI extremely high (overbought in range)
            elif crsi[i] > 75.0:
                desired_signal = -SIZE
        
        # TREND REGIME (trend follow) - align with HTF
        elif is_trend:
            # Long when CRSI low + HTF bullish
            if crsi[i] < 45.0 and hma_1w_bull:
                desired_signal = SIZE
            # Short when CRSI high + HTF bearish
            elif crsi[i] > 55.0 and hma_1w_bear:
                desired_signal = -SIZE
        
        # NEUTRAL REGIME (38.2-61.8) - use CRSI extremes only
        else:
            if crsi[i] < 20.0:
                desired_signal = SIZE
            elif crsi[i] > 80.0:
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