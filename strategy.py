#!/usr/bin/env python3
"""
Experiment #202: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + CRSI + Donchian

Hypothesis: 12h timeframe provides optimal balance between signal quality and trade frequency.
Using proven patterns from research:
1. Choppiness Index regime detection (CHOP>55=range, CHOP<45=trend)
2. Connors RSI for mean reversion entries in ranging markets
3. Donchian breakout for trend entries in trending markets
4. 1d KAMA for directional bias, 1w HMA for macro regime
5. Asymmetric sizing: full size with HTF trend, half size counter-trend

Key innovations from #199:
1. Higher TF (12h vs 4h) = fewer trades, lower fee drag, better signal quality
2. Dual HTF (1d+1w) for stronger regime confirmation
3. Donchian breakout added for trending regime (CRSI alone misses trends)
4. Looser CRSI thresholds (20/80 instead of 15/85) for more trades
5. Hold logic: stay in position while regime valid, exit on regime change

TARGET: 25-40 trades/year on 12h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_donchian_chop_regime_1d1w_v1"
timeframe = "12h"
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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    kama_14 = calculate_kama(close, er_period=10)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1d KAMA for directional bias (aligned properly)
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 1w HMA for macro regime (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(kama_14[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF MACRO BIAS ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 55.0  # Ranging market
        is_trend = chop_14[i] < 45.0  # Trending market
        # Neutral zone 45-55: hold current bias, use trend logic
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (CRSI extremes)
            # Long: CRSI < 20 (oversold) + price above 1w HMA (macro bullish) or neutral
            if crsi[i] < 20:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_FULL  # With macro trend
                elif price_above_kama_1d:
                    new_signal = POSITION_SIZE_HALF  # With 1d trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Counter-trend, smaller size
            
            # Short: CRSI > 80 (overbought) + price below 1w HMA (macro bearish) or neutral
            elif crsi[i] > 80:
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_FULL  # With macro trend
                elif price_below_kama_1d:
                    new_signal = -POSITION_SIZE_HALF  # With 1d trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Counter-trend, smaller size
        
        elif is_trend:
            # TREND FOLLOWING MODE (Donchian breakout + KAMA filter)
            # Long: Price breaks Donchian upper + price above KAMA(14)
            if close[i] >= donchian_upper[i-1] and close[i] > kama_14[i]:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_FULL  # With macro trend
                elif price_above_kama_1d:
                    new_signal = POSITION_SIZE_HALF  # With 1d trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Counter-trend, smaller size
            
            # Short: Price breaks Donchian lower + price below KAMA(14)
            elif close[i] <= donchian_lower[i-1] and close[i] < kama_14[i]:
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_FULL  # With macro trend
                elif price_below_kama_1d:
                    new_signal = -POSITION_SIZE_HALF  # With 1d trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Counter-trend, smaller size
        
        else:
            # NEUTRAL ZONE (45-55 CHOP): Use hybrid approach
            # CRSI mean reversion with KAMA trend filter
            if crsi[i] < 25 and close[i] > kama_14[i]:
                new_signal = POSITION_SIZE_HALF
            elif crsi[i] > 75 and close[i] < kama_14[i]:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought yet AND price above KAMA
                if crsi[i] < 75 and close[i] > kama_14[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold yet AND price below KAMA
                if crsi[i] > 25 and close[i] < kama_14[i]:
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
        # Exit long if price crosses below 1d KAMA (trend changed)
        if in_position and position_side > 0 and price_below_kama_1d:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d KAMA (trend changed)
        if in_position and position_side < 0 and price_above_kama_1d:
            new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit mean reversion positions if regime switches to strong trend
        if in_position and is_trend and chop_14[i] < 38.2:
            # In strong trend, mean reversion positions are risky
            if position_side > 0 and crsi[i] > 50:
                new_signal = 0.0
            elif position_side < 0 and crsi[i] < 50:
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