#!/usr/bin/env python3
"""
Experiment #036: 12h Primary + 1d HTF — Simplified Connors RSI Regime Switch

Hypothesis: Learning from 32 failed experiments, especially the 0-trade failures (#025, #027, #029, #033, #035):
1. Funding rate dependency causes 0 trades — REMOVE it entirely
2. Entry conditions too strict — RELAX thresholds (CRSI 20/80 vs 15/85)
3. Too many regime filters — SIMPLIFY to just CHOP + CRSI + HTF trend
4. 12h timeframe has shown promise (#026 +13.3%, #032 +5.3%) — use it properly
5. Connors RSI worked for ETH (Sharpe +0.923 in research) — use proven formula

Key changes from failed experiments:
- NO funding rate filter (causes 0 trades when data unavailable)
- CRSI thresholds: 20/80 (not 15/85) — ensures trades generate
- CHOP thresholds: 50/60 (not 45/55) — wider regime bands
- 1d HMA trend: only require price vs HMA (not multiple HTF alignment)
- Size: 0.25-0.30 discrete levels

Entry Logic:
- CHOPPY (CHOP>55): CRSI<20 long, CRSI>80 short (mean reversion)
- TRENDING (CHOP<50): Price>1d_HMA long bias, Price<1d_HMA short bias + CRSI confirmation
- Size: 0.30 with HTF alignment, 0.25 without

Risk: 2.5x ATR trailing stop, max signal 0.35, discrete levels
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_hma_1d_simplified_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - proven mean reversion indicator
    CRSI = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    
    More sensitive than regular RSI, catches reversals faster.
    Thresholds: <20 = oversold (long), >80 = overbought (short)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI(Streak, 2) - streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total < 1e-10:
            rsi_streak[i] = 50.0
        else:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / (total + 1e-10)
    
    # PercentRank(100) - percentage of prior closes lower than current
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_lower / rank_period
    
    # Combine into CRSI
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if np.isnan(rsi_close[i]) or np.isnan(rsi_streak[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator for HTF"""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - regime detection
    CHOP > 61.8 = range/choppy (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    We use 50/60 thresholds for more trade generation
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high <= lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        ratio = atr_sum / (highest_high - lowest_low)
        if ratio <= 0:
            chop[i] = 100.0
        else:
            chop[i] = 100.0 * np.log10(ratio) / np.log10(period)
    
    return chop

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
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.25
    MAX_SIZE = 0.35
    
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
        if np.isnan(hma_1d_aligned[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Wider thresholds to ensure trades generate
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 50.0
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - use CRSI extremes
            # RELAXED thresholds: 20/80 instead of 15/85 to ensure trades
            if crsi[i] < 20.0:
                # Oversold - long bias
                if price_above_hma_1d:
                    desired_signal = BASE_SIZE  # aligned with HTF
                else:
                    desired_signal = REDUCED_SIZE  # counter HTF
            elif crsi[i] > 80.0:
                # Overbought - short bias
                if price_below_hma_1d:
                    desired_signal = -BASE_SIZE  # aligned with HTF
                else:
                    desired_signal = -REDUCED_SIZE  # counter HTF
        
        elif is_trending:
            # TREND REGIME - trade with HTF trend + CRSI confirmation
            if price_above_hma_1d:
                # Bullish trend - look for pullback longs
                if crsi[i] < 40.0:  # Pullback in uptrend
                    desired_signal = BASE_SIZE
                elif crsi[i] > 70.0:  # Overbought in uptrend - reduce or flat
                    desired_signal = 0.0
                else:
                    desired_signal = REDUCED_SIZE  # Hold long
            elif price_below_hma_1d:
                # Bearish trend - look for rally shorts
                if crsi[i] > 60.0:  # Rally in downtrend
                    desired_signal = -BASE_SIZE
                elif crsi[i] < 30.0:  # Oversold in downtrend - reduce or flat
                    desired_signal = 0.0
                else:
                    desired_signal = -REDUCED_SIZE  # Hold short
        
        else:
            # NEUTRAL REGIME (50 <= CHOP <= 55) - only trade with HTF trend
            if price_above_hma_1d and crsi[i] < 50.0:
                desired_signal = REDUCED_SIZE
            elif price_below_hma_1d and crsi[i] > 50.0:
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
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        elif abs(desired_signal) >= 0.15:
            final_signal = np.sign(desired_signal) * REDUCED_SIZE
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