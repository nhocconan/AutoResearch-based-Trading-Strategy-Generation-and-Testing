#!/usr/bin/env python3
"""
Experiment #610: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion + Choppiness Regime

Hypothesis: 1h timeframe with Connors RSI (CRSI) provides superior mean-reversion entries
during range/bear markets (2022 crash, 2025 test). CRSI combines 3 components for more
reliable extreme detection than standard RSI. Combined with Choppiness for regime filter
and 4h/1d HMA for trend bias, this should generate fewer but higher-quality trades.

Key differences from failed strategies:
1. CRSI (not RSI) - 3-component oscillator with 75% win rate at extremes
2. Session filter (08-20 UTC) - avoids Asian session whipsaws
3. Very strict entry confluence: CRSI extreme + CHOP regime + HTF alignment
4. Target 40-80 trades/year (not 200+) to minimize fee drag
5. 1h entries timed to 4h trend direction (proven MTF pattern)

Strategy logic:
1. 1d HMA(21) = macro trend bias
2. 4h HMA(21) = medium trend bias
3. 1h CRSI(3,2,100) = entry trigger (CRSI<10 long, CRSI>90 short)
4. 1h Choppiness(14) = regime (CHOP>55 = range, CHOP<45 = trend)
5. Session filter: only trade 08-20 UTC (high volume)
6. ATR(14)*2.5 stoploss on all positions

Entry confluence (ALL must agree):
- LONG: CRSI<15 + price>4h_HMA + price>1d_HMA + CHOP>50 OR (CHOP<45 + 4h_HMA sloping up)
- SHORT: CRSI>85 + price<4h_HMA + price<1d_HMA + CHOP>50 OR (CHOP<45 + 4h_HMA sloping down)

Target: Sharpe>0.40, trades>=40 train (10/year), trades>=5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h1d_session_v1"
timeframe = "1h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    
    Entry signals: CRSI<10 = oversold (long), CRSI>90 = overbought (short)
    """
    n = len(close)
    if n < rank_period + rsi_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_streak_gain[i] / (avg_streak_loss[i] + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank of price change
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(rank_period, n):
        current_change = close[i] - close[i-1] if i > 0 else 0
        changes = np.diff(close[i-rank_period:i+1])
        if len(changes) > 0:
            pct_rank[i] = 100.0 * np.sum(changes < current_change) / len(changes)
        else:
            pct_rank[i] = 50.0
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
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

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # Convert ms to seconds, then to datetime
        ts_sec = open_time_array[i] / 1000.0
        hours[i] = int((ts_sec % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract UTC hour for session filter
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
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
        
        # === SESSION FILTER (08-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === HTF BIAS (1d macro + 4h medium) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # 4h HMA slope (5-bar lookback)
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-5] if i >= 5 and not np.isnan(hma_4h_aligned[i-5]) else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-5] if i >= 5 and not np.isnan(hma_4h_aligned[i-5]) else False
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean revert)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        chop_neutral = not chop_range and not chop_trend
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # CRSI recovery (turning up from oversold)
        crsi_turning_up = crsi[i] > crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        crsi_turning_down = crsi[i] < crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at CRSI extremes (primary strategy)
        if chop_range and in_session:
            # Long: CRSI oversold + price above 4h HMA (bullish bias in range)
            if crsi_oversold and htf_bull:
                desired_signal = SIZE_BASE
            # Strong long: extreme oversold + HTF bull + CRSI turning up
            elif crsi_extreme_oversold and htf_bull and crsi_turning_up:
                desired_signal = SIZE_STRONG
            # Short: CRSI overbought + price below 4h HMA (bearish bias in range)
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE_BASE
            # Strong short: extreme overbought + HTF bear + CRSI turning down
            elif crsi_extreme_overbought and htf_bear and crsi_turning_down:
                desired_signal = -SIZE_STRONG
        
        # TREND REGIME: Follow HTF direction on CRSI pullbacks
        elif chop_trend and in_session:
            # Long in uptrend: CRSI pullback + 4h HMA sloping up
            if crsi_oversold and htf_bull and hma_4h_slope_bull:
                desired_signal = SIZE_BASE
            # Short in downtrend: CRSI bounce + 4h HMA sloping down
            elif crsi_overbought and htf_bear and hma_4h_slope_bear:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Only take extreme CRSI signals with HTF confirmation
        elif chop_neutral and in_session:
            if crsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            elif crsi_extreme_overbought and htf_bear:
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