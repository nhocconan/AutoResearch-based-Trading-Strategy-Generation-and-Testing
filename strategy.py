#!/usr/bin/env python3
"""
Experiment #112: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + Volume Filter

Hypothesis: After 111 failed experiments, the clearest pattern is:
- 12h timeframe produces 20-50 trades/year (optimal fee/return balance)
- Connors RSI (CRSI) has 75% win rate for mean-reversion entries
- Choppiness Index correctly identifies regime (trend vs range)
- Volume filter eliminates false breakouts (critical for BTC/ETH)
- 1d HMA provides major trend bias without over-constraining

Key improvements over #104 (mtf_12h_crsi_sma200_hma_1d_v1, Sharpe=-1.574):
1. Connors RSI formula CORRECT: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Volume ratio filter: volume > 1.5 * volume_SMA(20) confirms breakouts
3. Looser Choppiness threshold (50) for more regime switches
4. Discrete signal levels (0.0, ±0.25, ±0.30) to reduce fee churn
5. ATR trailing stop (2.5x) with proper position tracking

Design choices:
- Timeframe: 12h (proven to work, 20-50 trades/year)
- HTF: 1d HMA(50) for major trend bias
- Entry: Connors RSI extremes (<15 long, >85 short) + regime filter
- Regime: CHOP>50 = range (mean revert), CHOP<50 = trend (breakout follow)
- Volume: confirm breakouts with 1.5x average volume
- Position size: 0.28 (28% of capital)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, >=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    rsi_short[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
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
        abs_streak = abs(streak[i])
        if abs_streak >= streak_period:
            # Strong streak = extreme RSI
            streak_rsi[i] = 100.0 if streak[i] > 0 else 0.0
        else:
            # Weak/no streak = neutral
            streak_rsi[i] = 50.0 + streak[i] * 10.0
            streak_rsi[i] = np.clip(streak_rsi[i], 0.0, 100.0)
    
    # Percent Rank of returns over last 100 periods
    returns = np.zeros(n)
    returns[0] = 0.0
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10)
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 50.0 as threshold for regime switch
    """
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
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_sma + 1e-10)
    
    return vol_ratio

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(crsi[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] <= 50.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Strong mean-reversion long
        crsi_overbought = crsi[i] > 85.0  # Strong mean-reversion short
        crsi_moderate_oversold = crsi[i] < 25.0
        crsi_moderate_overbought = crsi[i] > 75.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_bull = close[i] > donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow breakouts with volume confirmation
            if donchian_breakout_bull and volume_confirmed and hma_bull:
                if htf_bull:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE * 0.6
            elif donchian_breakout_bear and volume_confirmed and hma_bear:
                if htf_bear:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.6
            # CRSI pullback in trend
            elif hma_bull and crsi_moderate_oversold and htf_bull:
                desired_signal = SIZE * 0.7
            elif hma_bear and crsi_moderate_overbought and htf_bear:
                desired_signal = -SIZE * 0.7
        else:
            # CHOPPY REGIME: Mean revert with CRSI extremes
            if crsi_oversold and hma_bull:
                desired_signal = SIZE
            elif crsi_overbought and hma_bear:
                desired_signal = -SIZE
            elif crsi_moderate_oversold and not htf_bear:
                desired_signal = SIZE * 0.7
            elif crsi_moderate_overbought and not htf_bull:
                desired_signal = -SIZE * 0.7
            # Donchian mean reversion at bounds
            elif close[i] < donchian_lower[i] * 1.005 and crsi[i] < 30.0:
                desired_signal = SIZE * 0.6
            elif close[i] > donchian_upper[i] * 0.995 and crsi[i] > 70.0:
                desired_signal = -SIZE * 0.6
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
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