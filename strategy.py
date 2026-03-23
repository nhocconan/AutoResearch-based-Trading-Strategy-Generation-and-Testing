#!/usr/bin/env python3
"""
Experiment #207: 1d Primary + 1w HTF — Dual Regime (Mean Revert + Trend) + CRSI

Hypothesis: Daily timeframe naturally limits trade frequency (20-50/year target).
Key insight from failures: #200 had 0 trades due to overly strict confluence.
This experiment SIMPLIFIES entry logic while keeping regime intelligence:

1. Choppiness Index (CHOP) regime detection: >55 = range, <45 = trend
2. Connors RSI (CRSI) for mean reversion timing in ranges (thresholds 25/75 for more trades)
3. KAMA slope for trend direction (simpler than crossover)
4. 1w HTF for macro bias (asymmetric sizing: full with trend, half against)
5. ATR trailing stoploss (2.5x) for risk management

Key differences from #199:
1. Primary TF = 1d (not 4h) → fewer trades, less fee drag
2. Looser CRSI thresholds (25/75 vs 15/85) → more trade opportunities
3. KAMA slope instead of price vs KAMA → smoother trend signal
4. Hold logic simplified → hold while regime valid, exit on regime change
5. Minimum 30 trades/year target (looser than #199's strict filters)

TARGET: 30-50 trades/year on 1d, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_kama_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/101):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.zeros(n)
    
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i-er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_kama_slope(kama, lookback=5):
    """Calculate KAMA slope (directional bias)."""
    n = len(kama)
    slope = np.zeros(n)
    for i in range(lookback, n):
        slope[i] = kama[i] - kama[i-lookback]
    return slope

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1] if j > 0 else high[j] - close[j]), abs(low[j] - close[j-1] if j > 0 else low[j] - close[j]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Long: CRSI < 25 (oversold - loosened from 15 for more trades)
    Short: CRSI > 75 (overbought - loosened from 85 for more trades)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - fast momentum
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI of streak - consecutive up/down
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_positive = np.maximum(streak, 0)
    streak_negative = np.abs(np.minimum(streak, 0))
    
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_positive[i-streak_period+1:i+1])
        avg_loss = np.mean(streak_negative[i-streak_period+1:i+1])
        if avg_loss < 1e-10:
            streak_rsi[i] = 100.0
        else:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / (avg_loss + 1e-10)))
    
    # PercentRank - where current close ranks vs last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(max(3, streak_period, rank_period), n):
        crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    kama_14 = calculate_kama(close, er_period=10)
    kama_slope = calculate_kama_slope(kama_14, lookback=5)
    
    # Calculate 1w KAMA for macro trend (aligned properly)
    kama_1w_raw = calculate_kama(df_1w['close'].values, er_period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    kama_1w_slope = calculate_kama_slope(kama_1w_aligned, lookback=5)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(kama_14[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        
        # === HTF MACRO BIAS (1w KAMA slope) ===
        weekly_bullish = kama_1w_slope[i] > 0
        weekly_bearish = kama_1w_slope[i] < 0
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 55.0  # Ranging market → mean reversion
        is_trend = chop_14[i] < 45.0  # Trending market → trend follow
        # Neutral zone 45-55: hold current position
        
        # === KAMA SLOPE DIRECTION (1d) ===
        daily_bullish = kama_slope[i] > 0
        daily_bearish = kama_slope[i] < 0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (CRSI extremes)
            # Long: CRSI < 25 (oversold)
            if crsi[i] < 25:
                if weekly_bullish or not weekly_bearish:
                    new_signal = POSITION_SIZE_FULL  # With or neutral to weekly trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Counter weekly trend, smaller
            
            # Short: CRSI > 75 (overbought)
            elif crsi[i] > 75:
                if weekly_bearish or not weekly_bullish:
                    new_signal = -POSITION_SIZE_FULL  # With or neutral to weekly trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Counter weekly trend, smaller
        
        elif is_trend:
            # TREND FOLLOWING MODE (KAMA slope + CRSI filter)
            # Long: KAMA slope positive + CRSI not overbought (< 70)
            if daily_bullish and crsi[i] < 70:
                if weekly_bullish:
                    new_signal = POSITION_SIZE_FULL  # Aligned trends
                else:
                    new_signal = POSITION_SIZE_HALF  # Counter weekly, smaller
            
            # Short: KAMA slope negative + CRSI not oversold (> 30)
            elif daily_bearish and crsi[i] > 30:
                if weekly_bearish:
                    new_signal = -POSITION_SIZE_FULL  # Aligned trends
                else:
                    new_signal = -POSITION_SIZE_HALF  # Counter weekly, smaller
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime still valid (don't exit on minor signal changes)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not extremely overbought and regime not strongly bearish
                if crsi[i] < 85 and not (is_trend and daily_bearish):
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not extremely oversold and regime not strongly bullish
                if crsi[i] > 15 and not (is_trend and daily_bullish):
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime strongly contradicts position
        if in_position and position_side > 0:
            # Exit long if strong trend regime with bearish KAMA
            if is_trend and daily_bearish and crsi[i] > 50:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if strong trend regime with bullish KAMA
            if is_trend and daily_bullish and crsi[i] < 50:
                new_signal = 0.0
        
        # === WEEKLY TREND REVERSAL EXIT ===
        # Exit if weekly trend strongly reverses against position
        if in_position and position_side > 0 and weekly_bearish and kama_1w_slope[i] < -0.01 * close[i]:
            new_signal = 0.0
        
        if in_position and position_side < 0 and weekly_bullish and kama_1w_slope[i] > 0.01 * close[i]:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals