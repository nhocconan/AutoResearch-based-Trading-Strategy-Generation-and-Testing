#!/usr/bin/env python3
"""
Experiment #121: 4h Primary + 1d HTF — Connors RSI Mean Reversion + Trend Filter

Hypothesis: After 120+ experiments, the clearest pattern is:
- Complex regime filters (Choppiness, ADX, dual-regime) = 0 trades or negative Sharpe
- Connors RSI (CRSI) has documented 75% win rate in academic literature
- 4h timeframe with 1d trend bias has proven successful (current best Sharpe=0.351)
- VERY loose entry thresholds ensure trade generation on ALL symbols (BTC/ETH/SOL)

This strategy uses PROVEN mean reversion with trend filter:
1. 1d EMA21 = major trend bias (price above/below)
2. Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 = entry trigger
3. 4h Bollinger Band confirmation (price at band extreme)
4. ATR trailing stoploss (2.5x) for risk management
5. MINIMAL filters to ensure 30+ trades on train, 3+ on test

Connors RSI components:
- RSI(3): short-term momentum
- RSI_Streak(2): streak duration strength
- PercentRank(100): where current price ranks vs last 100 bars

Entry logic:
- LONG: 1d bull + CRSI<15 + price<Bollinger Lower
- SHORT: 1d bear + CRSI>85 + price>Bollinger Upper

Why this works:
- CRSI captures oversold/overbought better than standard RSI
- 1d trend filter prevents counter-trend mean reversion in strong trends
- BB confirmation ensures price at statistical extreme
- Loose CRSI thresholds (15/85 vs 10/90) ensure trades generate

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
Position size: 0.28 (28% of capital, conservative for 4h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_bb_trend_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors Relative Strength Index (CRSI)
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term price momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Percentile rank of price change over 100 periods
    
    Returns values 0-100. Extreme <10 = oversold, >90 = overbought.
    We use <15/>85 for looser entries to ensure trade generation.
    """
    n = len(close)
    if n < rank_period + 10:
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
    
    rsi_close = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on absolute streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    for i in range(streak_period, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total < 1e-10:
            rsi_streak[i] = 50.0
        else:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / total
    
    # Component 3: PercentRank of price change over rank_period
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        changes = np.diff(close[i-rank_period:i+1])
        if len(changes) > 0 and np.std(changes) > 1e-10:
            current_change = close[i] - close[i-1]
            pct_rank[i] = 100.0 * np.sum(changes < current_change) / len(changes)
        else:
            pct_rank[i] = 50.0
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion confirmation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

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

def calculate_ema(close, period=21):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d EMA21 for major trend bias
    ema_1d_raw = calculate_ema(df_1d['close'].values, period=21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # Calculate primary (4h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d EMA21) ===
        htf_bull = close[i] > ema_1d_aligned[i]
        htf_bear = close[i] < ema_1d_aligned[i]
        
        # === CRSI EXTREMES (Mean Reversion Signal) ===
        crsi_oversold = crsi[i] < 15.0  # Loose threshold for trade generation
        crsi_overbought = crsi[i] > 85.0
        
        # === BOLLINGER BAND CONFIRMATION ===
        at_lower_band = close[i] <= bb_lower[i] * 1.001  # At or below lower band
        at_upper_band = close[i] >= bb_upper[i] * 0.999  # At or above upper band
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + CRSI oversold + price at lower BB
        # SHORT: 1d bear + CRSI overbought + price at upper BB
        desired_signal = 0.0
        
        if htf_bull and crsi_oversold and at_lower_band:
            desired_signal = SIZE
        elif htf_bear and crsi_overbought and at_upper_band:
            desired_signal = -SIZE
        
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