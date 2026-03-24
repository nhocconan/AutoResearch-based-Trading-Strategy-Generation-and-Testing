#!/usr/bin/env python3
"""
Experiment #067: 1d Primary + 1w HTF — Choppiness Index Regime + Connors RSI

Hypothesis: Daily timeframe with Choppiness Index regime detection + Connors RSI 
for entry timing will work better than 4h strategies for BTC/ETH in bear/range markets.

Key differences from failed experiments:
1. Primary TF = 1d (fewer trades, less fee drag, proven to work better)
2. Choppiness Index (CHOP) for regime: >61.8 = range (mean revert), <38.2 = trend
3. Connors RSI (CRSI) for entries: combines RSI(3) + RSI_Streak(2) + PercentRank(100)
4. 1w HTF HMA for long-term bias filter
5. Looser CRSI thresholds (20/80 not 10/90) to ensure trade generation
6. ATR trailing stop at 3.0x for protection

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    def wma(data, span):
        res = np.full(len(data), np.nan)
        w = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            res[i] = np.sum(data[i - span + 1:i + 1] * w) / np.sum(w)
        return res
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    double_wma = 2.0 * wma_half - wma_full
    hma = wma(double_wma, sqrt_p)
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * (SUM(ATR, period) / (Highest High - Lowest Low)) / (log10(period))
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    hh_ll = hh - ll
    
    # Choppiness Index
    chop = np.full(n, np.nan)
    log_period = np.log10(period)
    for i in range(period, n):
        if hh_ll[i] > 1e-10:
            chop[i] = 100.0 * (atr_sum[i] / hh_ll[i]) / log_period
        else:
            chop[i] = 100.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Short-term momentum
    RSI(Streak): Consecutive up/down days
    PercentRank: Where current price ranks vs last 100 days
    
    CRSI < 20 = oversold (long opportunity)
    CRSI > 80 = overbought (short opportunity)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - short term momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI(Streak) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # PercentRank - where current price ranks vs last rank_period days
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi

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
    """Donchian Channel - breakout levels"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
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
    
    # Calculate and align 1w HMA for HTF trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_fast = calculate_hma(close, period=10)
    hma_slow = calculate_hma(close, period=30)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # Larger size for trend trades
    SIZE_MR = 0.25     # Slightly smaller for mean reversion
    
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
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
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
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA crossover) ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # === REGIME (Choppiness Index) ===
        is_ranging = chop[i] > 55.0  # Slightly lower threshold for more range detection
        is_trending = chop[i] < 45.0  # Slightly higher threshold for more trend detection
        # Neutral zone (45-55): use HTF bias only
        
        # === CONNORS RSI (Mean Reversion) ===
        crsi_oversold = crsi[i] < 25.0  # Looser for trade generation
        crsi_overbought = crsi[i] > 75.0  # Looser for trade generation
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_bull = close[i] > donch_upper[i-1]  # Break above previous high
        donch_breakout_bear = close[i] < donch_lower[i-1]  # Break below previous low
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND FOLLOWING PATH (CHOP < 45)
        if is_trending:
            # Long: HTF bull + HMA cross bull + Donchian breakout
            if hma_1w_bull and hma_cross_bull and donch_breakout_bull:
                desired_signal = SIZE_TREND
            # Short: HTF bear + HMA cross bear + Donchian breakout
            elif hma_1w_bear and hma_cross_bear and donch_breakout_bear:
                desired_signal = -SIZE_TREND
        
        # MEAN REVERSION PATH (CHOP > 55)
        elif is_ranging:
            # Long: HTF bull bias + CRSI oversold
            if hma_1w_bull and crsi_oversold:
                desired_signal = SIZE_MR
            # Short: HTF bear bias + CRSI overbought
            elif hma_1w_bear and crsi_overbought:
                desired_signal = -SIZE_MR
        
        # NEUTRAL ZONE (45 <= CHOP <= 55): Use HTF bias + CRSI only
        else:
            # Long: HTF bull + CRSI very oversold
            if hma_1w_bull and crsi[i] < 20.0:
                desired_signal = SIZE_MR
            # Short: HTF bear + CRSI very overbought
            elif hma_1w_bear and crsi[i] > 80.0:
                desired_signal = -SIZE_MR
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
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