#!/usr/bin/env python3
"""
Experiment #113: 5m Primary + 15m/4h HTF — Connors RSI + HMA Trend + Session Filter

Hypothesis: 5m timeframe is unexplored (0 experiments). Key insights:
- 5m requires EXTREME selectivity to avoid fee drag (target 50-120 trades/year)
- Session filter MANDATORY: only trade during active crypto hours (08-20 UTC)
- NEVER trade counter-trend: use 4h HMA for major direction bias
- 15m RSI for momentum confirmation, 5m Connors RSI for precise entry timing
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 20 + price > 4h HMA + 15m RSI > 40 (pullback in uptrend)
- Short: CRSI > 80 + price < 4h HMA + 15m RSI < 60 (rally in downtrend)
- Position size: 0.18 (smaller for 5m due to more trades = fee drag)
- Stoploss: 2.5x ATR trailing stop

Why this might work:
- 4h HMA filters out counter-trend trades (major cause of losses in 2022/2025)
- Connors RSI has 75% win rate on pullbacks (proven in literature)
- Session filter avoids low-liquidity hours (reduces slippage/whipsaws)
- 15m momentum confirmation adds confluence without being too restrictive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_crsi_hma_session_15m4h_v1"
timeframe = "5m"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    RSI(Streak): RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak: consecutive up/down days
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # PercentRank(100): percentage of prior closes lower than current
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_lower / rank_period
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def is_active_session(open_time):
    """
    Check if bar is within active trading session (08:00-20:00 UTC)
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    hour = (open_time // 1000 // 3600) % 24
    return 8 <= hour < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for major trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for momentum confirmation
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    rsi_5m = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (smaller for 5m due to fee drag)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_5m[i]) or np.isnan(crsi[i]) or np.isnan(rsi_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        if not is_active_session(open_time[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) - NEVER trade counter-trend ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM CONFIRMATION ===
        mom_bull = rsi_15m_aligned[i] > 40.0  # not oversold on 15m
        mom_bear = rsi_15m_aligned[i] < 60.0  # not overbought on 15m
        
        # === CONNORS RSI EXTREMES (entry timing) ===
        crsi_oversold = crsi[i] < 20.0  # extreme pullback
        crsi_overbought = crsi[i] > 80.0  # extreme rally
        
        # === 5m HMA TREND ===
        hma_bull = close[i] > hma_5m[i]
        hma_bear = close[i] < hma_5m[i]
        
        # === DESIRED SIGNAL (Trend + Pullback) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m momentum + CRSI oversold + 5m HMA bull
        if htf_bull and mom_bull and crsi_oversold and hma_bull:
            desired_signal = SIZE
        # SHORT: 4h bear + 15m momentum + CRSI overbought + 5m HMA bear
        elif htf_bear and mom_bear and crsi_overbought and hma_bear:
            desired_signal = -SIZE
        # Fallback: less strict CRSI but still with HTF bias
        elif htf_bull and mom_bull and crsi[i] < 30.0 and rsi_5m[i] < 40.0:
            desired_signal = SIZE * 0.7
        elif htf_bear and mom_bear and crsi[i] > 70.0 and rsi_5m[i] > 60.0:
            desired_signal = -SIZE * 0.7
        
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