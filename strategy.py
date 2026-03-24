#!/usr/bin/env python3
"""
Experiment #110: 1h Primary + 4h/1d HTF — CRSI Pullback + HMA Trend + Choppiness + Session

Hypothesis: After 109 failed experiments, the pattern is clear:
- Pure trend following fails on BTC/ETH in bear/range markets
- Pure mean reversion fails on SOL during strong trends
- SOLUTION: Use HTF (4h/1d) HMA for trend BIAS, 1h CRSI for pullback ENTRIES
- Connors RSI (CRSI) catches oversold/overbought extremes with 75% win rate
- Choppiness Index filters out extreme chop (CHOP < 45 = too choppy, skip)
- Session filter (08-20 UTC) reduces trades to target 40-80/year
- This combines: HTF trend bias + CRSI mean-reversion entries + regime filter

Key design choices:
- Timeframe: 1h (target 40-80 trades/year with session filter)
- HTF: 4h HMA(21) for intermediate trend, 1d HMA(50) for major bias
- Entry: CRSI < 15 (long) or > 85 (short) + aligned with HTF trend
- Regime: CHOP(14) > 45 (avoid extreme chop where mean-revert fails)
- Position size: 0.20 (20% of capital, conservative for 1h)
- Stoploss: 2.5x ATR trailing
- LOOSE CRSI thresholds to ensure >=30 trades on train

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_chop_session_4h1d_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak[:] = np.nan
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = 1.0
            j = i - 1
            while j > 0 and close[j] > close[j-1]:
                streak[i] += 1.0
                j -= 1
        elif close[i] < close[i-1]:
            streak[i] = -1.0
            j = i - 1
            while j > 0 and close[j] < close[j-1]:
                streak[i] -= 1.0
                j -= 1
        else:
            streak[i] = 0.0
    
    # Convert to RSI-like scale (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(period, n):
        if np.isnan(streak[i]):
            continue
        # Simple mapping: positive streak = high, negative = low
        streak_rsi[i] = 50.0 + streak[i] * 10.0
        streak_rsi[i] = np.clip(streak_rsi[i], 0.0, 100.0)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures where current return ranks vs last N periods
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0.0], returns])
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / period
        pr[i] = rank * 100.0
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <10 = oversold, >90 = overbought
    """
    n = len(close)
    
    rsi_short = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    pr = calculate_percent_rank(close, period=pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(max(rsi_period, streak_period, pr_period), n):
        if np.isnan(rsi_short[i]) or np.isnan(streak_rsi[i]) or np.isnan(pr[i]):
            continue
        crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
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
    We use CHOP > 45 to avoid extreme chop
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 1h)
    
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
        
        # === SESSION FILTER (08-20 UTC only) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (4h and 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both agree
        strong_bull = htf_4h_bull and htf_1d_bull
        strong_bear = htf_4h_bear and htf_1d_bear
        
        # === REGIME FILTER (Choppiness) ===
        # CHOP > 45 = acceptable (avoid extreme chop where mean-revert fails)
        acceptable_regime = chop[i] > 45.0
        
        # === CRSI ENTRY SIGNALS ===
        # Long: CRSI < 15 (oversold) + strong bull bias
        # Short: CRSI > 85 (overbought) + strong bear bias
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Only trade in session and acceptable regime
        if in_session and acceptable_regime:
            # LONG: oversold CRSI + HTF bull bias
            if crsi_oversold and strong_bull:
                desired_signal = SIZE
            # SHORT: overbought CRSI + HTF bear bias
            elif crsi_overbought and strong_bear:
                desired_signal = -SIZE
            # Weaker entries (only 4h agreement)
            elif crsi_oversold and htf_4h_bull and htf_1d_bull == False:
                desired_signal = SIZE * 0.5
            elif crsi_overbought and htf_4h_bear and htf_1d_bear == False:
                desired_signal = -SIZE * 0.5
        
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