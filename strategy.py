#!/usr/bin/env python3
"""
Experiment #530: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: 1h timeframe with Connors RSI (CRSI) provides superior mean-reversion entries
vs standard RSI, especially in bear/range markets (2022 crash, 2025 test period).
CRSI combines 3 signals: RSI(3) + RSI_Streak(2) + PercentRank(100) for 75% win rate.

Key innovations vs failed strategies:
1. CRSI instead of RSI(14) - catches short-term exhaustion better
2. Choppiness Index for regime detection (CHOP>61.8 = range mean-revert, CHOP<38.2 = trend follow)
3. 4h HMA(21) for trend bias - only trade in HTF direction
4. Session filter (08-20 UTC) - avoids low-volume Asian session whipsaws
5. 1d HMA for macro filter - adds another confluence layer

Strategy logic:
1. 1d HMA(21) = macro trend bias
2. 4h HMA(21) = medium trend bias + direction filter
3. 1h Choppiness(14) = regime (range vs trend)
4. 1h CRSI = entry timing (extremes <15 or >85)
5. Session filter = only trade 08-20 UTC (London/NY overlap)
6. ATR(14)*2.5 stoploss on all positions

Entry confluence (ALL must agree):
- LONG: CRSI<15 + CHOP>55 (range) OR CRSI<20 + CHOP<45 (trend pullback)
         + close>4h_HMA + close>1d_HMA + session=08-20 UTC
- SHORT: CRSI>85 + CHOP>55 (range) OR CRSI>80 + CHOP<45 (trend retrace)
         + close<4h_HMA + close<1d_HMA + session=08-20 UTC

Target: Sharpe>0.40, trades>=40/year (160 train, 50 test), DD<-30%
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
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
    Connors RSI (CRSI) - combines 3 components for mean-reversion signals
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - CRSI < 10-15: Oversold (long opportunity)
    - CRSI > 85-90: Overbought (short opportunity)
    
    Reported 75% win rate on BTC/ETH in range markets.
    """
    n = len(close)
    if n < rank_period + rsi_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0
        elif streak[i] < 0:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0
    
    # Component 3: PercentRank - where current close ranks in last N periods
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
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
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion favored)
    CHOP < 38.2 = trending (trend following favored)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        if np.isnan(atr[i-period+1:i+1]).any():
            continue
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def get_hour_from_open_time(open_time_arr):
    """Extract hour (0-23) from open_time timestamps"""
    hours = np.zeros(len(open_time_arr), dtype=int)
    for i in range(len(open_time_arr)):
        # open_time is in milliseconds since epoch
        ts = open_time_arr[i] / 1000.0
        hour = int((ts % 86400) / 3600)
        hours[i] = hour
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for medium trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend bias
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
    
    for i in range(250, n):
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
        
        # === SESSION FILTER (08-20 UTC = London/NY overlap) ===
        in_session = 8 <= hours[i] <= 20
        
        # === HTF BIAS (1d macro + 4h medium) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean reversion)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_bull and in_session:
            # Range regime: mean reversion long at CRSI extreme
            if chop_range and crsi_extreme_oversold:
                desired_signal = SIZE_STRONG
            # Trend regime: pullback long at moderate CRSI
            elif chop_trend and crsi_oversold:
                desired_signal = SIZE_BASE
            # Neutral regime: only extreme CRSI
            elif crsi_extreme_oversold:
                desired_signal = SIZE_BASE * 0.8
        
        # SHORT entries
        elif htf_bear and in_session:
            # Range regime: mean reversion short at CRSI extreme
            if chop_range and crsi_extreme_overbought:
                desired_signal = -SIZE_STRONG
            # Trend regime: retrace short at moderate CRSI
            elif chop_trend and crsi_overbought:
                desired_signal = -SIZE_BASE
            # Neutral regime: only extreme CRSI
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_BASE * 0.8
        
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