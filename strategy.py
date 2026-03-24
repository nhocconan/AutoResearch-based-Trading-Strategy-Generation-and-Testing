#!/usr/bin/env python3
"""
Experiment #770: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: Bear/range markets (2025+) favor mean reversion over trend following.
Connors RSI (CRSI) has 75% win rate for reversals. Choppiness Index filters regime.
4h HMA provides directional bias to avoid counter-trend mean reversion.

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Choppiness Index(14) > 61.8 = range (enable mean reversion entries)
3. 4h HMA(21) for HTF trend bias — only mean revert WITH HTF trend
4. 1d HMA(21) for macro regime filter
5. Session filter: 08-20 UTC only (highest volume hours)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30

Entry conditions (balanced for 40-80 trades/year):
- LONG: 4h HMA bull + 1d HMA bull + CHOP>61.8 + CRSI<15 + session 08-20 UTC
- SHORT: 4h HMA bear + 1d HMA bear + CHOP>61.8 + CRSI>85 + session 08-20 UTC

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h1d_session_v1"
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

def calculate_crsi(close):
    """
    Connors RSI - combines short-term momentum, streak, and percentile rank
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like scale
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(2, n):
        abs_streak = abs(streak[i])
        if abs_streak >= 2:
            streak_rsi[i] = 100.0 if streak[i] > 0 else 0.0
        elif abs_streak == 1:
            streak_rsi[i] = 75.0 if streak[i] > 0 else 25.0
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100) - where current return ranks vs last 100 bars
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(100, n):
        returns = np.diff(close[i-99:i+1])
        current_return = returns[-1]
        rank = np.sum(returns[:-1] < current_return)
        percent_rank[i] = (rank / 99.0) * 100.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Max High - Min Low)) / LOG10(n)
    CHOP > 61.8 = range market, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = np.zeros(n)
    atr_sum[:] = np.nan
    for i in range(period, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Calculate highest high and lowest low over period
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(period, n):
        hh[i] = np.max(high[i-period+1:i+1])
        ll[i] = np.min(low[i-period+1:i+1])
    
    # Calculate CHOP
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        if atr_sum[i] > 1e-10 and (hh[i] - ll[i]) > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
    
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

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
    crsi = calculate_crsi(close)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 8) and (utc_hour <= 20)
        
        if not in_session:
            # Close position if out of session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === MACRO REGIME (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop[i] > 61.8  # Range market - enable mean reversion
        is_trend = chop[i] < 38.2  # Trending market - disable mean reversion
        
        # === CRSI MEAN REVERSION SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === ENTRY LOGIC (MEAN REVERSION WITH HTF BIAS) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 1d bull + range market + CRSI oversold
        if htf_4h_bull and htf_1d_bull and is_range:
            if crsi_oversold:
                if crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 1d bear + range market + CRSI overbought
        elif htf_4h_bear and htf_1d_bear and is_range:
            if crsi_overbought:
                if crsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                else:
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
                entry_atr = atr_14[i]
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