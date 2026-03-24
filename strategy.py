#!/usr/bin/env python3
"""
Experiment #058: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: After 57 experiments, the key insight is that 30m strategies fail due to:
1. Too many trades → fee drag (need 30-80 trades/year MAX)
2. Entry conditions too strict → 0 trades (experiments #050, #051, #052, #055)
3. No regime filter → trading mean-reversion in trends (and vice versa)

This strategy combines:
1. Choppiness Index (14) regime filter: CHOP > 55 = range (mean revert), CHOP < 45 = trend
2. 4h HMA(21) trend bias: only long if price > 4h HMA, only short if price < 4h HMA
3. Connors RSI (CRSI) for entry timing: CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. Session filter: only trade 8-20 UTC (high volume hours, 24 of 48 daily bars)
5. Volume filter: volume > 0.8x 20-bar average
6. Discrete sizing: 0.20 (smaller for 30m to reduce fee impact)
7. ATR(14) 2.5x trailing stoploss

Why this should work:
- CHOP regime prevents wrong strategy in wrong market (mean-revert in range, trend-follow in trend)
- 4h HMA ensures we trade with HTF trend (proven in experiment #056)
- CRSI < 30 / > 70 happens frequently enough for trades (unlike RSI < 20 / > 80)
- Session filter reduces noise during low-volume hours
- 30m entries within 4h trend = HTF frequency with LTF precision

Entry Logic:
- Long: CHOP > 55 (range) OR (CHOP < 45 + 4h HMA bull) + CRSI < 30 + session 8-20 UTC + volume ok
- Short: CHOP > 55 (range) OR (CHOP < 45 + 4h HMA bear) + CRSI > 70 + session 8-20 UTC + volume ok
- Size: 0.20 (discrete)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%, trades/year < 100
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_4h1d_hma_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = rangebound, CHOP < 38.2 = trending
    We use 55/45 as thresholds for regime detection
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
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines momentum, streak, and percentile rank
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): Consecutive up/down days momentum
    PercentRank(100): Where current price ranks vs last 100 bars
    
    Entry: CRSI < 20-30 = oversold (long), CRSI > 70-80 = overbought (short)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
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
    
    # RSI Streak (2) - consecutive up/down bars
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
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total < 1e-10:
            rsi_streak[i] = 50.0
        else:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / total
    
    # Percent Rank (100) - where does current close rank vs last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (30m) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume moving average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (8-20 UTC)
    hours = get_hour_from_open_time(open_time)
    session_ok = (hours >= 8) & (hours < 20)
    
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position size (smaller for 30m)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_range = chop[i] > 55.0  # Rangebound market
        chop_trend = chop[i] < 45.0  # Trending market
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === CRSI ENTRY SIGNALS (LOOSE thresholds for trades) ===
        crsi_oversold = crsi[i] < 30.0  # Long entry
        crsi_overbought = crsi[i] > 70.0  # Short entry
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === SESSION FILTER ===
        session_active = session_ok[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Range regime (mean revert) OR Trend regime + HTF bull + CRSI oversold
        # Must have session + volume filter
        if session_active and volume_ok:
            if chop_range and crsi_oversold:
                # Range market + oversold = mean reversion long
                desired_signal = SIZE
            elif chop_trend and hma_4h_bull and crsi_oversold:
                # Trend market + HTF bull + pullback = trend follow long
                desired_signal = SIZE
        
        # Short entry: Range regime (mean revert) OR Trend regime + HTF bear + CRSI overbought
        if session_active and volume_ok:
            if chop_range and crsi_overbought:
                # Range market + overbought = mean reversion short
                desired_signal = -SIZE
            elif chop_trend and hma_4h_bear and crsi_overbought:
                # Trend market + HTF bear + retracement = trend follow short
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