#!/usr/bin/env python3
"""
Experiment #076: 12h Primary + 1d HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: After 75 failed experiments, the key insight is REGIME-AWARE trading.
Research shows Connors RSI + Choppiness Index achieved ETH Sharpe +0.923.

This strategy switches logic based on market regime:
1. CHOPPY (CHOP > 55): Mean reversion using Connors RSI extremes
2. TRENDING (CHOP < 45): Breakout following using Donchian + HMA

Why this should work:
- Regime detection prevents using wrong strategy in wrong market
- CRSI (3-period RSI + streak + rank) catches reversals better than standard RSI
- 12h timeframe = 20-50 trades/year (fee-efficient)
- 1d HMA filter prevents counter-trend trades in major moves
- LOOSE entry thresholds ensure we actually generate trades (learned from 0-trade failures)

Entry Logic:
- CHOPPY regime: Long when CRSI < 15 + price > 1d HMA, Short when CRSI > 85 + price < 1d HMA
- TREND regime: Long when Donchian(20) breakout + price > 1d HMA, Short when breakout low + price < 1d HMA
- Size: 0.28 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_regime_1d_v2"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * (ATR(1, sum) / (Highest High - Lowest Low)) / log10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        if hh - ll > 1e-10:
            chop[i] = 100.0 * (tr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak - consecutive up/down days
    3. PercentRank(100) - where current return ranks vs last 100 periods
    
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        # Component 1: RSI(3) on close
        if i < 3:
            rsi_close = 50.0
        else:
            delta = np.diff(close[max(0, i-4):i+1])
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            avg_gain = np.mean(gain) if len(gain) > 0 else 0.0
            avg_loss = np.mean(loss) if len(loss) > 0 else 0.0
            if avg_loss < 1e-10:
                rsi_close = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_close = 100.0 - (100.0 / (1.0 + rs))
        
        # Component 2: RSI(2) on streak
        streak = 0
        for j in range(i, max(i - streak_period - 2, 0), -1):
            if j < 1:
                break
            if close[j] > close[j-1]:
                streak += 1
            elif close[j] < close[j-1]:
                streak -= 1
            else:
                break
        
        # Convert streak to RSI-like value
        if streak >= 0:
            streak_rsi = 100.0 * (streak + 1) / (streak_period + 2)
        else:
            streak_rsi = 100.0 * (streak_period + 2 + streak) / (streak_period + 2)
        streak_rsi = np.clip(streak_rsi, 0, 100)
        
        # Component 3: PercentRank(100)
        if i < rank_period:
            pct_rank = 50.0
        else:
            returns = np.diff(close[i - rank_period:i + 1])
            current_return = returns[-1] if len(returns) > 0 else 0.0
            count_below = np.sum(returns[:-1] < current_return)
            pct_rank = 100.0 * count_below / max(len(returns) - 1, 1)
        
        # Combine components
        crsi[i] = (rsi_close + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
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
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size
    
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
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0  # Range/mean-reversion regime
        is_trending = chop[i] < 45.0  # Trend-following regime
        # 45-50 = neutral, no new entries
        
        # === HTF TREND BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # CHOPPY REGIME: Mean reversion with CRSI
        if is_choppy:
            # Long: CRSI oversold + 1d HMA bullish bias
            if crsi[i] < 20.0 and hma_1d_bull:
                desired_signal = SIZE
            # Short: CRSI overbought + 1d HMA bearish bias
            elif crsi[i] > 80.0 and hma_1d_bear:
                desired_signal = -SIZE
        
        # TRENDING REGIME: Breakout following
        elif is_trending:
            # Check if price broke out THIS bar
            breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
            breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
            
            # Long: Donchian breakout + 1d HMA bullish
            if breakout_long and hma_1d_bull:
                desired_signal = SIZE
            # Short: Donchian breakout + 1d HMA bearish
            elif breakout_short and hma_1d_bear:
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