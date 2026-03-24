#!/usr/bin/env python3
"""
Experiment #818: 4h Primary + 1d HTF — Connors RSI + Choppiness Dual Regime

Hypothesis: 4h timeframe with 1d HTF bias + Connors RSI entries + Choppiness regime
filter will outperform simple HMA/RSI strategies. Connors RSI (CRSI) has proven
75% win rate on mean reversion. Choppiness Index distinguishes range vs trend
regimes for adaptive entry logic.

Key innovations:
1. 1d HMA(21) for HTF trend bias — simple, reliable direction filter
2. 4h Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. Choppiness Index(14) regime: >61.8 = range (mean revert), <38.2 = trend
4. Dual entry logic: mean revert in chop, trend pullback in trend
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥30 trades/train, ≥3/test):
- LONG in CHOP: CRSI<20 + price>1d_HMA
- LONG in TREND: CRSI<40 + price>1d_HMA + HMA16>HMA48
- SHORT in CHOP: CRSI>80 + price<1d_HMA
- SHORT in TREND: CRSI>60 + price<1d_HMA + HMA16<HMA48

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d_v1"
timeframe = "4h"
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
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of current close vs last N closes
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measure consecutive up/down days
    streak = np.zeros(n)
    direction = np.zeros(n)  # 1=up, -1=down, 0=neutral
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            direction[i] = 1
        elif close[i] < close[i-1]:
            direction[i] = -1
        else:
            direction[i] = 0
    
    # Calculate streak length
    for i in range(1, n):
        if direction[i] == direction[i-1] and direction[i] != 0:
            streak[i] = streak[i-1] + 1
        elif direction[i] != 0:
            streak[i] = 1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (longer streak = more extreme)
    # Max streak of 10+ gets RSI of 100 or 0
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if streak[i] >= streak_period:
            if direction[i] > 0:
                rsi_streak[i] = min(100.0, 50.0 + streak[i] * 10.0)
            else:
                rsi_streak[i] = max(0.0, 50.0 - streak[i] * 10.0)
        else:
            rsi_streak[i] = 50.0  # neutral
    
    # Percent Rank - where does current close rank vs last N closes?
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
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate Choppiness Index
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        range_hl = highest[i] - lowest[i]
        if range_hl > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / range_hl) / np.log10(period)
    
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_4h_bull = hma_16[i] > hma_48[i]
        hma_4h_bear = hma_16[i] < hma_48[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Slightly lower threshold for more regime switches
        is_trending = chop[i] < 45.0  # Slightly higher threshold for more regime switches
        
        # === CONNORS RSI CONDITIONS (LOOSE for more trades) ===
        crsi_oversold = crsi[i] < 30.0  # Mean reversion long
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 70.0  # Mean reversion short
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # CRSI pullback levels for trend following
        crsi_pullback_long = crsi[i] < 45.0
        crsi_pullback_short = crsi[i] > 55.0
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        desired_signal = 0.0
        
        if htf_1d_bull:
            # LONG bias from HTF
            if is_choppy:
                # Mean reversion in choppy market
                if crsi_oversold:
                    if crsi_extreme_oversold:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            elif is_trending:
                # Trend pullback in trending market
                if crsi_pullback_long and hma_4h_bull:
                    if crsi_extreme_oversold:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            else:
                # Neutral regime - use either logic
                if crsi_oversold or (crsi_pullback_long and hma_4h_bull):
                    if crsi_extreme_oversold:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
        
        elif htf_1d_bear:
            # SHORT bias from HTF
            if is_choppy:
                # Mean reversion in choppy market
                if crsi_overbought:
                    if crsi_extreme_overbought:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            elif is_trending:
                # Trend pullback in trending market
                if crsi_pullback_short and hma_4h_bear:
                    if crsi_extreme_overbought:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            else:
                # Neutral regime - use either logic
                if crsi_overbought or (crsi_pullback_short and hma_4h_bear):
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