#!/usr/bin/env python3
"""
Experiment #539: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: 1h timeframe with Connors RSI (cRSI) provides superior mean-reversion entries
vs standard RSI. cRSI combines RSI(3) + RSI_Streak(2) + PercentRank(100) for 75% win rate.
Combined with Choppiness Index for regime detection and 4h/12h HMA for trend bias,
this should work in both bull and bear markets. Session filter (08-20 UTC) reduces
noise during low-volume Asian session.

Key differences from failed #530 (mtf_1h_crsi_chop_session_4h1d_v1):
1. Simpler entry logic - cRSI extremes only (no conflicting filters)
2. Dual HTF: 4h HMA for trend + 12h HMA for macro bias
3. Looser cRSI thresholds (20/80 instead of 10/90) to ensure trades generate
4. Session filter applied as bonus multiplier, not hard gate
5. Reduced position size churn (discrete levels only)

Strategy logic:
1. 12h HMA(21) = macro trend bias (very slow filter)
2. 4h HMA(21) = medium trend bias
3. 1h Choppiness(14) = regime (CHOP>61.8 = range, CHOP<38.2 = trend)
4. 1h Connors RSI = entry timing (cRSI<20 long, cRSI>80 short)
5. Session filter: 08-20 UTC gets 1.2x size bonus
6. ATR(14)*2.5 stoploss on all positions

Regime-adaptive entries:
- RANGE (CHOP>55): Mean revert at cRSI extremes with HTF confirmation
- TREND (CHOP<45): Follow HTF direction on cRSI pullback
- TRANSITION: Reduced size or flat

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=15 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_4h12h_v2"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean-reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Very fast RSI for short-term extremes
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current price ranks in last 100 bars
    
    Long: CRSI < 20 (oversold)
    Short: CRSI > 80 (overbought)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_fast = np.zeros(n)
    rsi_fast[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_fast[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_fast[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_delta = np.diff(streak)
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
    
    # Component 3: PercentRank (where current close ranks in last rank_period bars)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine all 3 components
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for medium trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h macro + 4h medium) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_12h_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean revert)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        is_range_regime = chop_range
        is_trend_regime = chop_trend
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # CRSI recovery (turning up from oversold)
        crsi_recovering_long = crsi_oversold and i > 0 and crsi[i] > crsi[i-1]
        crsi_recovering_short = crsi_overbought and i > 0 and crsi[i] < crsi[i-1]
        
        # === SESSION FILTER (08-20 UTC = London/NY overlap) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        is_peak_session = 8 <= hour_utc <= 20
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at CRSI extremes (primary signal)
        if is_range_regime:
            # Long: CRSI extreme oversold
            if crsi_extreme_oversold:
                desired_signal = SIZE_BASE
            # Short: CRSI extreme overbought
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_BASE
            # Long: CRSI recovering from oversold + HTF neutral/bull
            elif crsi_recovering_long and not htf_bear:
                desired_signal = SIZE_HALF
            # Short: CRSI recovering from overbought + HTF neutral/bear
            elif crsi_recovering_short and not htf_bull:
                desired_signal = -SIZE_HALF
        
        # TREND REGIME: Follow HTF direction on CRSI pullback
        elif is_trend_regime:
            # Long: HTF bull + CRSI pullback (not extreme)
            if htf_bull and crsi_oversold and crsi[i] > 15.0:
                desired_signal = SIZE_STRONG
            # Short: HTF bear + CRSI pullback (not extreme)
            elif htf_bear and crsi_overbought and crsi[i] < 85.0:
                desired_signal = -SIZE_STRONG
        
        # NEUTRAL/TRANSITION: Only take extreme CRSI signals
        else:
            if crsi_extreme_oversold:
                desired_signal = SIZE_HALF
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_HALF
        
        # === SESSION BONUS (increase size during peak hours) ===
        if is_peak_session and desired_signal != 0.0:
            if desired_signal > 0:
                desired_signal = min(desired_signal * 1.2, SIZE_STRONG)
            else:
                desired_signal = max(desired_signal * 1.2, -SIZE_STRONG)
        
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
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
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