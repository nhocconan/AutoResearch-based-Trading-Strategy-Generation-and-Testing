#!/usr/bin/env python3
"""
Experiment #1663: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + Donchian

Hypothesis: Previous 1d strategies worked but can be improved. This combines:
1. Connors RSI (CRSI) - proven 75% win rate for mean reversion in bear markets
2. Choppiness Index regime switch - CHOP>61.8=range, CHOP<38.2=trend
3. 1w HMA for long-term trend bias (slower, more reliable than 1d)
4. Donchian(20) breakout confirmation for trend regime
5. ATR trailing stop for risk management

Why this should beat Sharpe=0.618:
- CRSI captures reversals better than RSI in 2022 crash and 2025 bear
- Regime switching avoids trend-following whipsaw in choppy markets
- 1w HTF provides stronger trend filter than 1d
- LOOSE thresholds ensure 30-60 trades/year target (not 0 trades like #1655-1661)

Key learnings from failures:
- #1655, #1656, #1657, #1659, #1660, #1661 all got Sharpe=0.000 (NO TRADES)
- Over-filtering kills strategies. Use thresholds that actually trigger.
- 1d timeframe with 1w HTF worked in #1653 (Sharpe=0.260)
- Current best #1662 uses 12h but only got Sharpe=0.025

Entry Logic:
- CHOPPY (CHOP > 55): CRSI < 15 long, CRSI > 85 short (mean revert)
- TRENDING (CHOP < 45): Donchian breakout + 1w HMA bias (trend follow)
- NEUTRAL (45-55): Only trade WITH 1w trend, reduced size

Risk: 2.5x ATR trailing stop, discrete signals (0.0, ±0.25, ±0.35)
Target: Sharpe > 0.618, trades > 30/symbol train, > 5/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Composite of 3 components
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10-15, Short when CRSI > 85-90
    Proven 75% win rate for mean reversion
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rsi_short[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period + 1, n):
        if streak_loss_smooth[i-1] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[i-1] / streak_loss_smooth[i-1]))
    
    # Component 3: Percent Rank of 1-day returns over 100 days
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        if np.all(np.isnan(window)):
            continue
        current = returns[i]
        count_below = np.sum(window < current)
        percent_rank[i] = count_below / rank_period * 100.0
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother, less lag than EMA"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.35
    REDUCED_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === 1w HMA BIAS ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - use CRSI extremes (LOOSE for trade generation)
            # Long: CRSI < 15 (extremely oversold)
            if crsi[i] < 15.0:
                signal_strength = BASE_SIZE if weekly_bull else REDUCED_SIZE
                desired_signal = signal_strength
            # Short: CRSI > 85 (extremely overbought)
            elif crsi[i] > 85.0:
                signal_strength = BASE_SIZE if weekly_bear else REDUCED_SIZE
                desired_signal = -signal_strength
        
        elif is_trending:
            # TREND REGIME - use Donchian breakout + 1w bias
            # Long: Price breaks Donchian upper + weekly bullish
            if close[i] >= donchian_upper[i] * 0.999:  # small tolerance
                signal_strength = BASE_SIZE if weekly_bull else REDUCED_SIZE
                desired_signal = signal_strength
            # Short: Price breaks Donchian lower + weekly bearish
            elif close[i] <= donchian_lower[i] * 1.001:
                signal_strength = BASE_SIZE if weekly_bear else REDUCED_SIZE
                desired_signal = -signal_strength
        
        else:
            # NEUTRAL REGIME (45 <= CHOP <= 55) - only trade WITH 1w trend
            # Use CRSI for entry timing
            if weekly_bull and crsi[i] < 30.0:
                desired_signal = REDUCED_SIZE
            elif weekly_bear and crsi[i] > 70.0:
                desired_signal = -REDUCED_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
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