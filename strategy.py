#!/usr/bin/env python3
"""
Experiment #120: 1h Primary + 4h/12h HTF — Connors RSI Pullback + Session Filter

Hypothesis: After 119 failed experiments, the pattern is clear:
- Lower TF (1h/30m) MUST have strict filters to avoid fee drag (>100 trades/yr = death)
- HTF trend (4h/12h) + LTF entry timing (1h CRSI pullback) = proven pattern
- Session filter (8-20 UTC) captures London/NY overlap = higher volume moves
- Connors RSI (CRSI) = (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
  - CRSI<15 = extreme oversold (long entry in uptrend)
  - CRSI>85 = extreme overbought (short entry in downtrend)
- Choppiness Index regime: CHOP>55 = range (mean revert), CHOP<45 = trend (follow)
- Volume filter: must be >0.7x 20-bar avg (avoid low liquidity entries)

Key design:
- Timeframe: 1h (30-60 trades/year target)
- HTF: 4h HMA for trend, 12h HMA for major bias
- Entry: CRSI extreme pullback WITHIN HTF trend
- Filters: Session 8-20 UTC, Volume >0.7x avg, CHOP regime
- Size: 0.25 (moderate for 1h, ensures trade generation)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.351, DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_session_chop_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    wma_half = pd.Series(close).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma

def calculate_crsi(close, rsi_period=2, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(2)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100 - (100 / (1 + rs))
    
    # Percent Rank (100)
    for i in range(pr_period, n):
        lookback = close[i-pr_period:i+1]
        current_change = close[i] - close[i-1] if i > 0 else 0
        changes = np.diff(lookback)
        if len(changes) > 0:
            rank = np.sum(changes < current_change) / len(changes) * 100
        else:
            rank = 50
        crsi[i] = (rsi[i] + rsi_streak[i] + rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            chop[i] = 100
            continue
        
        tr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    crsi = calculate_crsi(close, rsi_period=2, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (moderate for 1h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Check if indicators are ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Session filter: 8-20 UTC (London/NY overlap)
        hour = (open_time[i] // 3600000) % 24
        if hour < 8 or hour >= 20:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Volume filter: must be > 0.7x average
        if volume[i] < 0.7 * vol_avg[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # HTF trend bias
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_12h_aligned[i]
        
        # Regime detection via Choppiness
        is_trending = chop[i] < 50
        is_ranging = chop[i] >= 50
        
        # === DESIRED SIGNAL ===
        # LONG: HTF bull + CRSI extreme oversold
        # SHORT: HTF bear + CRSI extreme overbought
        desired_signal = 0.0
        
        if htf_bull:
            # Trending: stricter CRSI threshold
            if is_trending and crsi[i] < 15:
                desired_signal = SIZE
            # Ranging: looser CRSI threshold (mean revert)
            elif is_ranging and crsi[i] < 25:
                desired_signal = SIZE
        elif htf_bear:
            # Trending: stricter CRSI threshold
            if is_trending and crsi[i] > 85:
                desired_signal = -SIZE
            # Ranging: looser CRSI threshold (mean revert)
            elif is_ranging and crsi[i] > 75:
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