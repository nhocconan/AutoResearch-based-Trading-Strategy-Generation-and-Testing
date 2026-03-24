#!/usr/bin/env python3
"""
Experiment #656: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + HTF HMA

Hypothesis: Adaptive regime detection via Choppiness Index + Connors RSI mean reversion
should outperform pure trend strategies in mixed 2021-2024 markets. Key insight:
- CHOP > 61.8 = range market → mean revert (CRSI extremes)
- CHOP < 38.2 = trend market → trend follow (HTF HMA direction)
- 30m for entry timing, 4h/1d for signal direction

Why this should work:
1. Choppiness Index filters out whipsaw periods (2022 crash was very choppy)
2. Connors RSI has 75% win rate on mean reversion entries
3. HTF HMA prevents fighting the major trend
4. Session filter (08-20 UTC) avoids low-liquidity traps
5. Conservative size (0.20-0.25) limits drawdown

Target: Sharpe>0.40, trades>=40 train, trades>=4 test, DD>-30%
Timeframe: 30m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_hma_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies range vs trend markets
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.nanmax(high[i-period+1:i+1])
        ll = np.nanmin(low[i-period+1:i+1])
        
        if hh > ll:
            # Sum of ATR over period
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                if j >= 1:
                    tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                else:
                    tr = high[j] - low[j]
                atr_sum += tr
            
            if atr_sum > 1e-10:
                choppiness[i] = 100.0 * (atr_sum / (hh - ll)) / np.sqrt(period)
    
    return choppiness

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # Calculate RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    # Calculate RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period + 1, n):
        streak = 0
        if close[i] > close[i-1]:
            streak = 1
            j = i - 1
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            streak = -1
            j = i - 1
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        
        # Convert streak to RSI-like value
        if streak > 0:
            streak_rsi[i] = 100.0 * streak / (streak + 1)
        elif streak < 0:
            streak_rsi[i] = 100.0 * abs(streak) / (abs(streak) + 1)
        else:
            streak_rsi[i] = 50.0
    
    # Calculate Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_rsi(close, period=14):
    """Standard RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

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
    
    # Calculate 30m indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
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
        hour = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour <= 20
        
        # === HTF BIAS (4h and 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment (both 4h and 1d agree)
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === CHOPPINESS REGIME ===
        is_range = choppiness[i] > 55.0  # Slightly lower threshold for more trades
        is_trend = choppiness[i] < 45.0  # Slightly higher threshold for more trades
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 25.0  # Looser threshold for more trades
        crsi_overbought = crsi[i] > 75.0  # Looser threshold for more trades
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC (ADAPTIVE BY REGIME) ===
        desired_signal = 0.0
        
        if in_session:
            # RANGE REGIME: Mean reversion
            if is_range:
                # LONG: HTF not strongly bear + CRSI oversold + RSI confirmation
                if not htf_strong_bear and crsi_oversold and rsi_oversold:
                    if htf_strong_bull:
                        desired_signal = SIZE_STRONG
                    elif htf_4h_bull:
                        desired_signal = SIZE_BASE
                    else:
                        desired_signal = SIZE_BASE * 0.5
                
                # SHORT: HTF not strongly bull + CRSI overbought + RSI confirmation
                elif not htf_strong_bull and crsi_overbought and rsi_overbought:
                    if htf_strong_bear:
                        desired_signal = -SIZE_STRONG
                    elif htf_4h_bear:
                        desired_signal = -SIZE_BASE
                    else:
                        desired_signal = -SIZE_BASE * 0.5
            
            # TREND REGIME: Trend following
            elif is_trend:
                # LONG: HTF bull + pullback (CRSI not overbought)
                if htf_strong_bull and not crsi_overbought:
                    if crsi[i] < 50.0:  # Pullback entry
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
                
                # SHORT: HTF bear + pullback (CRSI not oversold)
                elif htf_strong_bear and not crsi_oversold:
                    if crsi[i] > 50.0:  # Pullback entry
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            
            # NEUTRAL REGIME: Only strong HTF signals
            else:
                if htf_strong_bull and crsi_oversold:
                    desired_signal = SIZE_BASE
                elif htf_strong_bear and crsi_overbought:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
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