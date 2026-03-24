#!/usr/bin/env python3
"""
Experiment #064: 4h Primary + 12h/1d HTF — KAMA + Connors RSI + Choppiness Regime

Hypothesis: Combining KAMA (adaptive trend) with Connors RSI (proven mean reversion)
and Choppiness Index regime filter will outperform simple ADX regime switching.

Key differences from #061:
1. KAMA instead of HMA - adapts to market efficiency/volatility automatically
2. Connors RSI (3-period RSI + streak RSI + percentile rank) - proven 75% win rate
3. Choppiness Index with 50 threshold (not 61.8/38.2) - cleaner regime split
4. Looser entry conditions to ensure 30+ trades/symbol on train
5. Asymmetric sizing: 0.30 trend, 0.25 mean reversion

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_crsi_chop_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = streak[i - 1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        abs_streak = abs(streak[i])
        if streak[i] >= 0:
            streak_rsi[i] = min(100.0, 50.0 + abs_streak * 10.0)
        else:
            streak_rsi[i] = max(0.0, 50.0 - abs_streak * 10.0)
    
    # Percentile Rank (100-period)
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        rank = np.sum(window[:-1] < close[i])
        pct_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Connors RSI
    crsi = (rsi + streak_rsi + pct_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market chop vs trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest:
            tr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

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

def calculate_donchian(high, low, period=20):
    """Donchian Channels - breakout detection"""
    n = len(close)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, mid, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h/1d KAMA for HTF trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, er_period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    kama_slow = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_MR = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h and 1d KAMA) ===
        htf_bull = close[i] > kama_12h_aligned[i] and close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_12h_aligned[i] and close[i] < kama_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === 4h TREND (KAMA crossover) ===
        kama_cross_bull = kama_fast[i] > kama_slow[i]
        kama_cross_bear = kama_fast[i] < kama_slow[i]
        
        # === REGIME (Choppiness Index) ===
        is_choppy = chop[i] > 50.0  # Range market
        is_trending = chop[i] <= 50.0  # Trend market
        
        # === CONNORS RSI MEAN REVERSION ===
        crsi_oversold = crsi[i] < 20.0  # Very oversold
        crsi_overbought = crsi[i] > 80.0  # Very overbought
        
        # === DONCHIAN BREAKOUT ===
        breakout_up = close[i] > donch_upper[i - 1] if not np.isnan(donch_upper[i - 1]) else False
        breakout_down = close[i] < donch_lower[i - 1] if not np.isnan(donch_lower[i - 1]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND FOLLOWING PATH (Chop <= 50)
        if is_trending:
            # Long: HTF bull + KAMA cross bull + Donchian breakout
            if htf_bull and kama_cross_bull and breakout_up:
                desired_signal = SIZE_TREND
            # Short: HTF bear + KAMA cross bear + Donchian breakout
            elif htf_bear and kama_cross_bear and breakout_down:
                desired_signal = -SIZE_TREND
            # Alternative: HTF neutral + strong KAMA cross
            elif htf_neutral and kama_cross_bull and close[i] > kama_slow[i] * 1.002:
                desired_signal = SIZE_TREND * 0.7
            elif htf_neutral and kama_cross_bear and close[i] < kama_slow[i] * 0.998:
                desired_signal = -SIZE_TREND * 0.7
        
        # MEAN REVERSION PATH (Chop > 50)
        if is_choppy:
            # Long: HTF not bear + CRSI oversold
            if not htf_bear and crsi_oversold:
                desired_signal = SIZE_MR
            # Short: HTF not bull + CRSI overbought
            elif not htf_bull and crsi_overbought:
                desired_signal = -SIZE_MR
            # Additional MR: price at Donchian extremes
            elif not htf_bear and close[i] < donch_lower[i - 1] * 1.005 if not np.isnan(donch_lower[i - 1]) else False:
                desired_signal = SIZE_MR * 0.8
            elif not htf_bull and close[i] > donch_upper[i - 1] * 0.995 if not np.isnan(donch_upper[i - 1]) else False:
                desired_signal = -SIZE_MR * 0.8
        
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
        if desired_signal >= SIZE_TREND * 0.85:
            final_signal = SIZE_TREND
        elif desired_signal >= SIZE_MR * 0.85:
            final_signal = SIZE_MR
        elif desired_signal <= -SIZE_TREND * 0.85:
            final_signal = -SIZE_TREND
        elif desired_signal <= -SIZE_MR * 0.85:
            final_signal = -SIZE_MR
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