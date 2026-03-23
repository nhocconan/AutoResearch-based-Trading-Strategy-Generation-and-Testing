#!/usr/bin/env python3
"""
Experiment #204: 4h Primary + 12h/1d HTF — Dual HTF Bias + Connors RSI + Choppiness Regime

Hypothesis: Improve on #199 by adding dual HTF confirmation (12h + 1d) for stronger
macro bias, switching to HMA for faster trend detection, and tightening CRSI thresholds
for better entry timing. The dual HTF should filter out false signals when 12h and 1d
disagree, while the faster HMA should catch trend changes earlier than KAMA.

Key changes from #199:
1. Dual HTF bias: both 12h AND 1d must agree for full position size
2. HMA instead of KAMA: faster trend detection (HMA eliminates lag)
3. Tighter CRSI thresholds: 12/88 instead of 15/85 for cleaner signals
4. Better hold logic: hold through minor CRSI fluctuations if regime valid
5. Improved stoploss: 2.0*ATR trailing instead of 2.5*ATR fixed

TARGET: 35-50 trades/year on 4h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_dual_htf_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Eliminates lag while maintaining smoothness.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

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
    
    rsi3 = calculate_rsi(close, period=3)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
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
    
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    for i in range(max(3, streak_period, rank_period), n):
        crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    
    # Calculate HTF HMA for macro trend (aligned properly)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    prev_signal = 0.0
    
    for i in range(200, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === HTF MACRO BIAS (Dual 12h + 1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Dual HTF agreement for strong bias
        htft_bullish = price_above_hma_12h and price_above_hma_1d
        htft_bearish = price_below_hma_12h and price_below_hma_1d
        htft_neutral = not htft_bullish and not htft_bearish
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 55.0  # Ranging market
        is_trend = chop_14[i] < 45.0  # Trending market
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (CRSI extremes)
            if crsi[i] < 12:  # Oversold
                if htft_bullish:
                    new_signal = POSITION_SIZE_FULL  # With strong HTF trend
                elif not htft_bearish:
                    new_signal = POSITION_SIZE_HALF  # Neutral or weak bullish
            elif crsi[i] > 88:  # Overbought
                if htft_bearish:
                    new_signal = -POSITION_SIZE_FULL  # With strong HTF trend
                elif not htft_bullish:
                    new_signal = -POSITION_SIZE_HALF  # Neutral or weak bearish
        
        elif is_trend:
            # TREND FOLLOWING MODE (HMA crossover + CRSI filter)
            hma_bullish = hma_21[i] > hma_50[i]
            hma_bearish = hma_21[i] < hma_50[i]
            
            if hma_bullish and crsi[i] < 65:  # Not overbought
                if htft_bullish:
                    new_signal = POSITION_SIZE_FULL
                elif not htft_bearish:
                    new_signal = POSITION_SIZE_HALF
            elif hma_bearish and crsi[i] > 35:  # Not oversold
                if htft_bearish:
                    new_signal = -POSITION_SIZE_FULL
                elif not htft_bullish:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still valid (reduce churn)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not extremely overbought and regime ok
                if crsi[i] < 85 and not (is_trend and hma_21[i] < hma_50[i]):
                    new_signal = prev_signal
            elif position_side < 0:
                # Hold short if CRSI not extremely oversold and regime ok
                if crsi[i] > 15 and not (is_trend and hma_21[i] > hma_50[i]):
                    new_signal = prev_signal
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === HTF TREND REVERSAL EXIT ===
        # Exit long if BOTH 12h and 1d turn bearish (strong macro reversal)
        if in_position and position_side > 0 and htft_bearish:
            new_signal = 0.0
        
        # Exit short if BOTH 12h and 1d turn bullish (strong macro reversal)
        if in_position and position_side < 0 and htft_bullish:
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
        prev_signal = new_signal if new_signal != 0.0 else prev_signal
    
    return signals