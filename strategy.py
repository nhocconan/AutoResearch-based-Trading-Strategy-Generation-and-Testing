#!/usr/bin/env python3
"""
Experiment #198: 30m Primary + 4h/1d HTF — Connors RSI + HMA Trend + Volatility Filter

Hypothesis: Lower timeframe (30m) strategies fail due to either (1) too many trades → fee drag,
or (2) too many filters → 0 trades. This strategy uses:
1. 4h HMA for trend direction (simple, proven)
2. Connors RSI (CRSI) for mean reversion entries (75% win rate in literature)
3. 1d HMA for macro bias filter
4. ATR volatility filter (skip extreme vol spikes)
5. LOOSE thresholds to ensure trade frequency (learned from 0-trade failures)

Key difference from failed 30m strategies:
- Fewer confluence requirements (3 instead of 5+)
- Looser CRSI thresholds (15/85 instead of 10/90)
- No session filter (kills too many trades)
- No volume spike requirement (inconsistent on crypto)

TARGET: 40-70 trades/year, Sharpe > 0.4 on ALL symbols
Position sizing: 0.0, ±0.25, ±0.30 (discrete levels)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_hma_volfilter_4h1d_v1"
timeframe = "30m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close prices - short-term momentum
    2. RSI(2) of streak - consecutive up/down days
    3. PercentRank(100) - where current price ranks vs last 100
    
    Long signal: CRSI < 15 (oversold)
    Short signal: CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # Component 1: RSI(3) of close
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0).values
    
    # Component 2: RSI(2) of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine components
    for i in range(max(rsi_period, streak_period, rank_period), n):
        crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (rate of change over lookback bars)."""
    n = len(hma_values)
    slope = np.zeros(n)
    for i in range(lookback, n):
        if hma_values[i-lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i-lookback]) / hma_values[i-lookback] * 100.0
        else:
            slope[i] = 0.0
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_slope = calculate_hma_slope(hma_21, lookback=5)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Calculate 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    hma_4h_slope = calculate_hma_slope(hma_4h_aligned, lookback=3)
    
    # Calculate 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    hma_1d_slope = calculate_hma_slope(hma_1d_aligned, lookback=2)
    
    # Volatility filter: ATR ratio (current vs 30-bar avg)
    atr_avg30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / (atr_avg30 + 1e-10)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_21[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_ratio[i]):
            continue
        
        # === HTF MACRO BIAS ===
        # 4h trend direction
        bullish_4h = close[i] > hma_4h_aligned[i] and hma_4h_slope[i] > -0.5
        bearish_4h = close[i] < hma_4h_aligned[i] and hma_4h_slope[i] < 0.5
        
        # 1d trend direction (stronger filter)
        bullish_1d = close[i] > hma_1d_aligned[i]
        bearish_1d = close[i] < hma_1d_aligned[i]
        
        # Combined bias
        strong_bullish = bullish_4h and bullish_1d
        strong_bearish = bearish_4h and bearish_1d
        neutral = not strong_bullish and not strong_bearish
        
        # === VOLATILITY FILTER ===
        # Skip extreme volatility (ATR ratio > 2.5 = panic/euphoria)
        vol_normal = atr_ratio[i] < 2.5
        
        # === ENTRY LOGIC (Connors RSI Mean Reversion) ===
        new_signal = 0.0
        
        if vol_normal:
            # LONG: CRSI < 15 (oversold) + bullish or neutral bias
            if crsi[i] < 15:
                if strong_bullish:
                    new_signal = POSITION_SIZE_FULL
                elif neutral:
                    new_signal = POSITION_SIZE_HALF
                # Skip if strong bearish (counter-trend risky)
            
            # SHORT: CRSI > 85 (overbought) + bearish or neutral bias
            elif crsi[i] > 85:
                if strong_bearish:
                    new_signal = -POSITION_SIZE_FULL
                elif neutral:
                    new_signal = -POSITION_SIZE_HALF
                # Skip if strong bullish
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and bias still valid (even if CRSI normalized)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h still bullish
                if bullish_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h still bearish
                if bearish_4h:
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
        # Exit long if 4h trend reverses bearish
        if in_position and position_side > 0 and bearish_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend reverses bullish
        if in_position and position_side < 0 and bullish_4h:
            new_signal = 0.0
        
        # === CRSI EXIT (take profit when normalized) ===
        # Exit long if CRSI > 70 (overbought after oversold entry)
        if in_position and position_side > 0 and crsi[i] > 70:
            new_signal = 0.0
        
        # Exit short if CRSI < 30 (oversold after overbought entry)
        if in_position and position_side < 0 and crsi[i] < 30:
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