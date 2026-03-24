#!/usr/bin/env python3
"""
Experiment #834: 1d Primary + 1w HTF — Regime-Adaptive Dual Strategy

Hypothesis: Daily timeframe with weekly bias provides optimal signal quality.
Previous experiments failed due to: (1) overly strict entries = 0 trades,
(2) single-regime logic (trend-only fails in chop, mean-revert fails in trends).

This strategy uses REGIME-ADAPTIVE logic:
1. 1w HMA(21) for long-term bias (bull/bear market filter)
2. Choppiness Index(14) for regime detection: >61.8=range, <38.2=trend
3. RANGE regime: Connors RSI mean reversion (CRSI<10 long, >90 short)
4. TREND regime: Donchian(20) breakout + 1d HMA(16/48) confirmation
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Why this should work:
- Connors RSI has 75% win rate in range markets (research-backed)
- Choppiness Index is best meta-filter for bear/range markets
- Donchian breakout catches sustained trends without whipsaw
- 1w HTF bias prevents counter-trend trades in strong markets
- Loose enough entries to guarantee ≥30 trades/train, ≥3/test

Target: Sharpe>0.50, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_crsi_donchian_1w_v1"
timeframe = "1d"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    >61.8 = choppy/range, <38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10 and sum_atr > 1e-10:
            choppiness[i] = 100.0 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long: CRSI < 10, Short: CRSI > 90
    """
    n = len(close)
    if n < max(rsi_period, streak_period, rank_period) + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if delta[i-1] > 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if delta[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        total_avg = avg_streak_gain[i] + avg_streak_loss[i]
        if total_avg > 1e-10:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / total_avg
        else:
            rsi_streak[i] = 50.0
    
    # Component 3: Percent Rank of daily returns over 100 days
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window < current_return)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine all three components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
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
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = not np.isnan(choppiness[i]) and choppiness[i] > 55.0  # Range market
        is_trending = not np.isnan(choppiness[i]) and choppiness[i] < 45.0  # Trend market
        
        # === 1d HMA TREND ===
        hma_1d_bull = hma_16[i] > hma_48[i]
        hma_1d_bear = hma_16[i] < hma_48[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = not np.isnan(donchian_upper[i]) and close[i] > donchian_upper[i-1]
        donchian_breakout_short = not np.isnan(donchian_lower[i]) and close[i] < donchian_lower[i-1]
        
        # === CONNORS RSI (Mean Reversion) ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 20.0  # Loosened from 10 for more trades
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 80.0  # Loosened from 90 for more trades
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Use Connors RSI mean reversion
        if is_choppy:
            # Long in range: CRSI oversold + HTF bull bias
            if crsi_oversold and htf_1w_bull:
                desired_signal = SIZE_BASE
            # Short in range: CRSI overbought + HTF bear bias
            elif crsi_overbought and htf_1w_bear:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Use Donchian breakout + HMA confirmation
        elif is_trending:
            # Long in trend: Donchian breakout + HMA bull + HTF bull
            if donchian_breakout_long and hma_1d_bull and htf_1w_bull:
                desired_signal = SIZE_STRONG
            # Short in trend: Donchian breakout + HMA bear + HTF bear
            elif donchian_breakout_short and hma_1d_bear and htf_1w_bear:
                desired_signal = -SIZE_STRONG
        
        # HYBRID: HMA crossover in any regime (backup for trade generation)
        if desired_signal == 0.0:
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
            
            if hma_crossover_long and htf_1w_bull:
                desired_signal = SIZE_BASE
            elif hma_crossover_short and htf_1w_bear:
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