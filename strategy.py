#!/usr/bin/env python3
"""
Experiment #090: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness + HMA Trend + Session

Hypothesis: After 89 failed experiments, the pattern for 1h timeframe is clear:
- Need VERY FEW trades (40-80/year) to avoid fee drag on lower TF
- Connors RSI (CRSI) proven 75% win rate for mean reversion entries
- Choppiness Index filters regime: CHOP>55 = range (mean revert), CHOP<55 = trend
- 4h HMA provides trend bias without being too restrictive
- Session filter (08-20 UTC) avoids low liquidity whipsaws
- This combines: CRSI (proven reversal) + CHOP (regime) + HTF HMA (bias) + Session (timing)

Key design choices:
- Timeframe: 1h (target 40-80 trades/year, strict entry filters)
- HTF: 4h HMA(21) for trend direction, 1d HMA(50) for major bias
- Entry: CRSI<15 (long) or CRSI>85 (short) + regime filter + session filter
- Regime: CHOP(14)>55 = range (mean revert), CHOP<55 = trend (pullback entries)
- Position size: 0.25 (25% of capital, conservative for 1h)
- Stoploss: 2.5x ATR trailing
- STRICT filters to ensure <=80 trades/year

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h1d_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Fast RSI for short-term momentum
    RSI_Streak(2): RSI of consecutive up/down streak length
    PercentRank(100): Percentile rank of price change over 100 periods
    
    Entry: CRSI < 10-15 (oversold) for long, CRSI > 85-90 (overbought) for short
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (use absolute streak values for RSI calculation)
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    
    # Percent Rank of price change over rank_period
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        price_changes = np.diff(close[i-rank_period:i+1])
        current_change = close[i] - close[i-1]
        rank = np.sum(price_changes < current_change)
        percent_rank[i] = 100.0 * rank / len(price_changes)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 55 as threshold for this strategy
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_session_filter(open_time):
    """
    Session filter: 08-20 UTC (high liquidity hours)
    Returns 1 during session, 0 outside
    """
    n = len(open_time)
    session = np.zeros(n)
    
    for i in range(n):
        # open_time is in milliseconds
        hour = (open_time[i] // 3600000) % 24
        if 8 <= hour < 20:
            session[i] = 1.0
        else:
            session[i] = 0.0
    
    return session

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    session = calculate_session_filter(open_time)
    
    # 1h HMA for additional trend confirmation
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h and 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === SESSION FILTER ===
        in_session = session[i] == 1.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Strong oversold
        crsi_overbought = crsi[i] > 85.0  # Strong overbought
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === 1h HMA TREND ===
        hma_1h_bull = close[i] > hma_1h[i]
        hma_1h_bear = close[i] < hma_1h[i]
        
        # === DESIRED SIGNAL (Dual Regime + CRSI) ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Mean reversion with CRSI extremes
            # LONG: CRSI extreme oversold + HTF not strongly bear + session
            if crsi_extreme_oversold and not htf_1d_bear and in_session:
                desired_signal = SIZE
            # SHORT: CRSI extreme overbought + HTF not strongly bull + session
            elif crsi_extreme_overbought and not htf_1d_bull and in_session:
                desired_signal = -SIZE
            # Weaker signal: CRSI oversold/overbought + 4h bias agreement
            elif crsi_oversold and htf_4h_bull and in_session:
                desired_signal = SIZE * 0.6
            elif crsi_overbought and htf_4h_bear and in_session:
                desired_signal = -SIZE * 0.6
        else:
            # TREND REGIME: Pullback entries with CRSI
            # LONG: pullback (CRSI oversold) + HTF bull + session
            if crsi_oversold and htf_4h_bull and htf_1d_bull and in_session:
                desired_signal = SIZE
            # SHORT: pullback (CRSI overbought) + HTF bear + session
            elif crsi_overbought and htf_4h_bear and htf_1d_bear and in_session:
                desired_signal = -SIZE
            # Weaker: CRSI extreme + 4h bias only
            elif crsi_extreme_oversold and htf_4h_bull and in_session:
                desired_signal = SIZE * 0.6
            elif crsi_extreme_overbought and htf_4h_bear and in_session:
                desired_signal = -SIZE * 0.6
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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