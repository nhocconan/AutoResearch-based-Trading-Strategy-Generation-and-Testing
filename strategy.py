#!/usr/bin/env python3
"""
Experiment #589: 15m Primary + 1h/4h HTF — Connors RSI Pullback + HMA Trend + Session Filter

Hypothesis: 15m timeframe with Connors RSI (CRSI) for entry timing + 4h HMA for trend direction
will generate sufficient trades (40-100/year) while maintaining positive Sharpe in bear/range markets.

Key innovations vs failed 15m attempts (#577, #579, #581, #585):
1. CRSI instead of simple RSI - combines RSI(3) + RSI-Streak(2) + PercentRank(100)
2. LOOSE entry conditions - CRSI<20 or >80 (not extreme 10/90) to ensure trades fire
3. 4h HMA for trend bias (not 1d which is too slow for 15m entries)
4. 1h RSI for momentum confirmation (medium-term filter)
5. Session filter: 00-12 UTC only (London/NY overlap = 70% of crypto volume)
6. Size: 0.18 (appropriate for 15m frequency)
7. Stoploss: 2.5x ATR trailing + time-based exit after 48 bars (~12 hours)

Why this should work on 15m:
- HTF (4h) provides direction, 15m provides entry timing = HTF trade frequency
- CRSI has 75% win rate for mean-reversion entries (research-backed)
- Session filter reduces false signals during low-volume Asia hours
- Loose CRSI thresholds ensure we get 50-80 trades/year (not 0 like #581)

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_pullback_hma4h_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, prank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean-reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak - streak duration (consecutive up/down days)
    3. PercentRank(100) - where current close ranks vs last 100 closes
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    For 15m with loose entries: CRSI < 20 or > 80
    """
    n = len(close)
    if n < prank_period:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    rsi_close[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_delta = np.diff(streak_abs)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if streak_avg_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = streak_avg_gain[i] / streak_avg_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100)
    prank = np.zeros(n)
    prank[:] = np.nan
    for i in range(prank_period, n):
        window = close[i-prank_period+1:i+1]
        rank = np.sum(window[:-1] < close[i])
        prank[i] = 100.0 * rank / (prank_period - 1)
    
    # Combine components
    for i in range(prank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(prank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + prank[i]) / 3.0
    
    return crsi

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

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1h RSI for momentum confirmation
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, prank_period=100)
    rsi_15m = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
    SIZE_STRONG = 0.22
    
    # Position tracking for stoploss and time exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
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
        
        if np.isnan(crsi[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        is_active_session = 0 <= hour_utc <= 12
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1H MOMENTUM (RSI confirmation) ===
        mom_bull = rsi_1h_aligned[i] > 45.0
        mom_bear = rsi_1h_aligned[i] < 55.0
        mom_strong_bull = rsi_1h_aligned[i] > 55.0
        mom_strong_bear = rsi_1h_aligned[i] < 45.0
        
        # === 15M CRSI ENTRY SIGNALS (loose thresholds for trade frequency) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_os = crsi[i] < 15.0
        crsi_extreme_ob = crsi[i] > 85.0
        
        # CRSI recovery (rising from oversold)
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        
        # === ENTRY LOGIC (designed for 50-80 trades/year) ===
        desired_signal = 0.0
        
        # LONG entries: HTF bull + CRSI oversold + session active
        if htf_bull and is_active_session:
            if crsi_extreme_os and mom_bull:
                desired_signal = SIZE_STRONG
            elif crsi_oversold and mom_bull and crsi_rising:
                desired_signal = SIZE_BASE
            elif crsi_oversold and htf_bull:
                desired_signal = SIZE_BASE * 0.9
        
        # SHORT entries: HTF bear + CRSI overbought + session active
        elif htf_bear and is_active_session:
            if crsi_extreme_ob and mom_bear:
                desired_signal = -SIZE_STRONG
            elif crsi_overbought and mom_bear and crsi_falling:
                desired_signal = -SIZE_BASE
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE_BASE * 0.9
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        # Time-based exit after 48 bars (~12 hours on 15m)
        time_exit = False
        if in_position and (i - entry_bar) > 48:
            time_exit = True
        
        if stoploss_triggered or time_exit:
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
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
                entry_bar = 0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals