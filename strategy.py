#!/usr/bin/env python3
"""
Experiment #541: 15m Primary + 4h/1d HTF — Connors RSI Mean Reversion + Trend Filter

Hypothesis: 15m timeframe with Connors RSI (CRSI) for mean reversion entries,
filtered by 4h trend direction. CRSI has 75% win rate in backtests and works
well in range/bear markets (2022 crash, 2025 bear). Simpler than complex regime
detection which caused 0 trades in recent 15m experiments (#529, #537).

Key differences from failed 15m attempts:
1. CRSI instead of standard RSI - combines RSI(3) + streak + rank for better signals
2. SINGLE HTF filter (4h HMA) not triple (1h+4h+1d) - reduces filter conflicts
3. Looser entry thresholds: CRSI < 15 (not < 10), CRSI > 85 (not > 90)
4. Session filter: 00-12 UTC only (London+NY overlap)
5. Position size: 0.15-0.20 (smaller for 15m frequency to reduce fee drag)

Strategy logic:
1. 4h HMA(21) = trend bias (long only when price > 4h HMA, short when <)
2. 1d HMA(50) = macro filter (avoid counter-trend trades)
3. 15m CRSI(3,2,100) = entry trigger (extreme oversold/overbought)
4. 15m SMA(200) = trend confirmation (long above, short below)
5. 15m ATR(14)*2.0 = stoploss
6. Session filter: only trade 00-12 UTC (high liquidity)

CRSI Formula:
- RSI(3): 3-period RSI for short-term momentum
- RSI_Streak(2): streak RSI (consecutive up/down days)
- PercentRank(100): rank of current price change vs last 100 bars
- CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

Entry conditions (LOOSENED for trade generation):
- Long: CRSI < 15 + price > 4h_HMA + price > SMA200 + session 00-12 UTC
- Short: CRSI > 85 + price < 4h_HMA + price < SMA200 + session 00-12 UTC

Target: Sharpe > 0.40, trades >= 100 train (25/year), trades >= 15 test
Timeframe: 15m
Position size: 0.15-0.20 (discrete levels to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_hma_4h1d_session_v1"
timeframe = "15m"
leverage = 1.0

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
    RSI Streak Component for Connors RSI
    Measures consecutive up/down periods
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak = 0
        if i > 0:
            # Count consecutive up or down moves
            if close[i] > close[i-1]:
                streak = 1
                for j in range(i-1, 0, -1):
                    if close[j] > close[j-1]:
                        streak += 1
                    else:
                        break
            elif close[i] < close[i-1]:
                streak = -1
                for j in range(i-1, 0, -1):
                    if close[j] < close[j-1]:
                        streak -= 1
                    else:
                        break
        
        # Convert streak to 0-100 scale
        # Positive streak = high RSI, negative = low RSI
        if streak >= period:
            streak_rsi[i] = 100.0
        elif streak <= -period:
            streak_rsi[i] = 0.0
        elif streak > 0:
            streak_rsi[i] = 50.0 + (streak / period) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 + (streak / period) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component for Connors RSI
    Ranks current price change vs last N periods
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(period, n):
        current_change = close[i] - close[i-1]
        count_higher = 0
        
        for j in range(i-period, i):
            if j > 0:
                past_change = close[j] - close[j-1]
                if current_change > past_change:
                    count_higher += 1
        
        percent_rank[i] = (count_higher / period) * 100.0
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Long signal: CRSI < 10-15 (oversold)
    Short signal: CRSI > 85-90 (overbought)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    rsi3 = calculate_rsi(close, period=rsi_period)
    streak = calculate_rsi_streak(close, period=streak_period)
    prank = calculate_percent_rank(close, period=rank_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak[i]) and not np.isnan(prank[i]):
            crsi[i] = (rsi3[i] + streak[i] + prank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_hma(close, period):
    """Hull Moving Average"""
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma200 = calculate_sma(close, period=200)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.18
    SIZE_SHORT = -0.18
    SIZE_REDUCE = 0.09  # Half position for take profit
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(sma200[i]):
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
        
        # === SESSION FILTER (00-12 UTC only) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === MACRO FILTER (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (SMA200) ===
        local_bull = close[i] > sma200[i]
        local_bear = close[i] < sma200[i]
        
        # === CRSI EXTREMES (LOOSENED for trade generation) ===
        crsi_oversold = crsi[i] < 15.0  # Was < 10
        crsi_overbought = crsi[i] > 85.0  # Was > 90
        
        # === ENTRY LOGIC (SIMPLIFIED) ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold + 4h bull + SMA200 bull + session
        if in_session and htf_bull and local_bull and crsi_oversold:
            desired_signal = SIZE_LONG
        
        # SHORT: CRSI overbought + 4h bear + SMA200 bear + session
        elif in_session and htf_bear and local_bear and crsi_overbought:
            desired_signal = SIZE_SHORT
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trailing stop
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trailing stop
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_LONG * 0.9:
            final_signal = SIZE_LONG
        elif desired_signal <= SIZE_SHORT * 0.9:
            final_signal = SIZE_SHORT
        elif abs(desired_signal) >= SIZE_LONG * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_REDUCE
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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