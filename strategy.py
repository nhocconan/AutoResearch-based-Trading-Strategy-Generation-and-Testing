#!/usr/bin/env python3
"""
Experiment #209: 4h Primary + 1d HTF — Volatility Regime + CRSI + KAMA Trend

Hypothesis: Switch between mean reversion and trend following based on volatility regime.
Low vol = range (fade CRSI extremes). High vol = trend (follow 1d KAMA with pullbacks).

Key innovations:
1. ATR ratio regime (ATR7/ATR30) instead of Choppiness - more responsive to vol changes
2. Connors RSI for precise mean reversion entries (proven 75% win rate)
3. 1d KAMA for macro trend bias (adaptive, less lag than EMA)
4. Asymmetric sizing: full size with trend, half size counter-trend
5. Looser CRSI thresholds (20/80) to ensure adequate trade frequency

TARGET: 30-50 trades/year on 4h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_regime_crsi_kama_1d_v1"
timeframe = "4h"
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    kama_14 = calculate_kama(close, er_period=10)
    
    # Calculate 1d KAMA for macro trend (aligned properly)
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Volatility regime: ATR ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        atr_ratio = atr_7 / (atr_30 + 1e-10)
    atr_ratio = np.nan_to_num(atr_ratio, nan=1.0)
    
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
        if np.isnan(crsi[i]):
            continue
        if np.isnan(kama_14[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(atr_ratio[i]):
            continue
        
        # === VOLATILITY REGIME ===
        # Low vol (ratio < 0.8) = range → mean reversion
        # High vol (ratio > 1.2) = trend → trend following
        # Neutral (0.8-1.2) = use recent bias
        is_low_vol = atr_ratio[i] < 0.8
        is_high_vol = atr_ratio[i] > 1.2
        
        # === HTF MACRO BIAS (1d KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_low_vol:
            # MEAN REVERSION MODE (CRSI extremes)
            # Long: CRSI < 20 (oversold)
            if crsi[i] < 20:
                if price_above_kama_1d:
                    new_signal = POSITION_SIZE_FULL  # With trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Counter-trend, smaller size
            
            # Short: CRSI > 80 (overbought)
            elif crsi[i] > 80:
                if price_below_kama_1d:
                    new_signal = -POSITION_SIZE_FULL  # With trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Counter-trend, smaller size
        
        elif is_high_vol:
            # TREND FOLLOWING MODE (KAMA + CRSI pullback)
            # Long: Price above KAMA(14) + CRSI pullback (30-60)
            if close[i] > kama_14[i] and 30 < crsi[i] < 65:
                if price_above_kama_1d:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: Price below KAMA(14) + CRSI pullback (35-70)
            elif close[i] < kama_14[i] and 35 < crsi[i] < 70:
                if price_below_kama_1d:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        else:
            # NEUTRAL REGIME - use simpler logic
            # Long: CRSI < 25 + price above 1d KAMA
            if crsi[i] < 25 and price_above_kama_1d:
                new_signal = POSITION_SIZE_HALF
            
            # Short: CRSI > 75 + price below 1d KAMA
            elif crsi[i] > 75 and price_below_kama_1d:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still valid (avoid churn)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not extremely overbought
                if crsi[i] < 85:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not extremely oversold
                if crsi[i] > 15:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1d KAMA (macro trend changed)
        if in_position and position_side > 0 and price_below_kama_1d:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d KAMA (macro trend changed)
        if in_position and position_side < 0 and price_above_kama_1d:
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