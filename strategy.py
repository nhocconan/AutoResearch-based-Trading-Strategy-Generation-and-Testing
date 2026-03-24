#!/usr/bin/env python3
"""
Experiment #419: 1h Primary + 4h/12h HTF — Simplified Confluence v1

Hypothesis: Recent failures (#410, #416, #417) had Sharpe=0.0 because entry conditions were too strict (0 trades).
This version SIMPLIFIES while keeping proven elements:
- 4h HMA for trend bias (direction)
- 1h CRSI for entry timing (mean reversion)
- Choppiness for regime filter
- Session filter (08-20 UTC) for liquidity

Key changes to ensure trades:
1. LOOSEN RSI thresholds (15/85 instead of 10/90)
2. Reduce confluence from 5+ to 3 filters max
3. Add fallback entry when regime unclear
4. Ensure stoploss triggers position exit

Target: 40-80 trades/year, Sharpe>0.4, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_chop_session_4h12h_v1"
timeframe = "1h"
leverage = 1.0

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
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        if streak_period > 0:
            streak_rsi[i] = 100.0 * up_streaks / streak_period
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100-period)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Combine
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 12h) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both HTF agree
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        
        # === REGIME (Choppiness) ===
        # CHOP > 61.8 = range (mean revert)
        # CHOP < 38.2 = trend (trend follow)
        is_choppy = chop[i] > 55.0  # Loosened from 61.8
        is_trending = chop[i] < 45.0  # Loosened from 38.2
        
        # === SESSION FILTER (08-20 UTC) ===
        # Get hour from open_time
        hour = pd.to_datetime(prices['open_time'].iloc[i], unit='ms').hour
        in_session = 8 <= hour <= 20
        
        # === CRSI EXTREMES (LOOSENED for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Was 10
        crsi_overbought = crsi[i] > 80.0  # Was 90
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - 3 filters max) ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion)
        if is_choppy:
            # Long: CRSI oversold + above SMA200 + (HTF neutral or bull)
            if crsi_oversold and above_sma200:
                if not htf_strong_bear:  # Don't short against strong HTF bear
                    desired_signal = SIZE_BASE
            
            # Short: CRSI overbought + below SMA200 + (HTF neutral or bear)
            elif crsi_overbought and below_sma200:
                if not htf_strong_bull:  # Don't long against strong HTF bull
                    desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (trend follow with pullback)
        elif is_trending:
            # Long: HTF bull + CRSI pullback (<50) + in session
            if htf_strong_bull and crsi[i] < 50.0 and in_session:
                desired_signal = SIZE_STRONG
            
            # Short: HTF bear + CRSI rally (>50) + in session
            elif htf_strong_bear and crsi[i] > 50.0 and in_session:
                desired_signal = -SIZE_STRONG
        
        # REGIME 3: NEUTRAL (fallback - use 4h HMA only)
        else:
            # Simple HMA cross with 4h bias
            hma_1h_bull = close[i] > hma_1h[i]
            hma_1h_bear = close[i] < hma_1h[i]
            
            if htf_4h_bull and hma_1h_bull and in_session:
                desired_signal = SIZE_BASE
            elif htf_4h_bear and hma_1h_bear and in_session:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * atr[i]
                else:
                    stop_price = entry_price + 2.5 * atr[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals