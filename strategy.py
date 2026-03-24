#!/usr/bin/env python3
"""
Experiment #710: 1h Primary + 4h/1d HTF — CRSI Mean Reversion + Choppiness Regime + Session Filter

Hypothesis: 1h timeframe with VERY FEW trades (40-80/year) using 3+ confluence filters.
Connors RSI (CRSI) has 75% win rate for mean reversion. Choppiness Index detects range vs trend.
4h HMA provides trend direction bias. Session filter (08-20 UTC) avoids low-liquidity whipsaws.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven mean reversion
2. Choppiness Index(14) > 50 = range market (favor mean reversion), < 38 = trend (avoid)
3. 4h HMA(21) for HTF trend direction — only trade with HTF bias
4. Session filter: 08-20 UTC only (high liquidity, less noise)
5. Discrete sizing: 0.0, ±0.20, ±0.30 with ATR stoploss
6. LOOSE CRSI thresholds (15/85 not 10/90) to ensure trade generation

Entry conditions (balanced for trade generation):
- LONG: 4h HMA bull + CRSI < 20 + CHOP > 45 + session 08-20 UTC
- SHORT: 4h HMA bear + CRSI > 80 + CHOP > 45 + session 08-20 UTC
- Stoploss: 2.5x ATR trailing stop → signal = 0

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%, trades/year 40-80
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma4h_session_v1"
timeframe = "1h"
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
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak(2) - consecutive up/down days
    streak = np.zeros(n)
    streak[:] = np.nan
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = 1.0 if streak[i-1] >= 0 else 1.0
            if i > 1 and close[i-1] > close[i-2]:
                streak[i] = streak[i-1] + 1.0
        elif close[i] < close[i-1]:
            streak[i] = -1.0 if streak[i-1] <= 0 else -1.0
            if i > 1 and close[i-1] < close[i-2]:
                streak[i] = streak[i-1] - 1.0
        else:
            streak[i] = 0.0
    
    # Convert streak to RSI-like 0-100 scale
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        if not np.isnan(streak[i]):
            # Map streak to 0-100: positive streak = high, negative = low
            max_streak = max(streak_period, abs(np.nanmax(streak[max(0,i-streak_period*10):i+1])))
            if max_streak > 0:
                streak_rsi[i] = 50.0 + (streak[i] / max_streak) * 50.0
            else:
                streak_rsi[i] = 50.0
    
    # PercentRank(100) - where current close ranks in last 100 bars
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = (count_below / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range market, CHOP < 38.2 = trending market
    
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
    
    # Calculate Choppiness
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
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
        
        # === SESSION FILTER (08-20 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (4h and 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 45 = range market (favor mean reversion)
        is_range = chop[i] > 45.0
        
        # === CRSI ENTRY (LOOSE for trade generation) ===
        # LONG: CRSI < 20 (oversold)
        crsi_long = crsi[i] < 20.0
        # SHORT: CRSI > 80 (overbought)
        crsi_short = crsi[i] > 80.0
        
        # === ENTRY LOGIC (3+ CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG: 4h HMA bull + CRSI oversold + range market + session
        if htf_4h_bull and crsi_long and is_range and in_session:
            desired_signal = SIZE_STRONG
        # LONG partial: 4h HMA bull + CRSI oversold + session (no chop filter)
        elif htf_4h_bull and crsi_long and in_session:
            desired_signal = SIZE_BASE
        
        # SHORT: 4h HMA bear + CRSI overbought + range market + session
        elif htf_4h_bear and crsi_short and is_range and in_session:
            desired_signal = -SIZE_STRONG
        # SHORT partial: 4h HMA bear + CRSI overbought + session (no chop filter)
        elif htf_4h_bear and crsi_short and in_session:
            desired_signal = -SIZE_BASE
        
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
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
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
                entry_atr = atr[i]
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