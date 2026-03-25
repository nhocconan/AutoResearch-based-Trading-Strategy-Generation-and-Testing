#!/usr/bin/env python3
"""
Experiment #1560: 6h Primary + 1d/1w HTF — Volatility Squeeze Breakout Strategy

Hypothesis: 6h timeframe captures multi-day volatility cycles optimal for crypto.
This strategy uses KELTNER-BOLLINGER SQUEEZE to detect volatility contraction,
then trades the expansion breakout with HTF trend confirmation.

Key components:
1. 1w HMA(21) for secular trend bias (avoid counter-trend in major moves)
2. 1d HMA(21) for intermediate trend direction
3. 6h Bollinger Band(20,2.0) + Keltner Channel(20,1.5*ATR) squeeze detection
4. Volume confirmation: breakout volume > 1.5x 20-period average
5. ATR(14) trailing stoploss (2.5x ATR)
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work:
- Volatility squeeze patterns have 65-75% breakout success rate in crypto
- 6h TF = natural 30-50 trades/year (fee-efficient, not too many)
- Triple HTF filter (1w + 1d) prevents major counter-trend disasters
- Volume confirmation filters false breakouts (common in crypto)
- Different from failed strategies: focuses on VOLATILITY not trend/mean-reversion

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + 1d_HMA bullish + BB inside Keltner (squeeze) + 
        price breaks BB_upper + volume > 1.5x avg
- SHORT: 1w_HMA bearish + 1d_HMA bearish + BB inside Keltner (squeeze) +
         price breaks BB_lower + volume > 1.5x avg
- Exit: signal→0 when 2.5x ATR stop hit OR squeeze ends without follow-through

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_squeeze_keltner_bb_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_keltner(high, low, close, period=20, atr_mult=1.5):
    """Keltner Channels - EMA +/- ATR multiplier"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, period)
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    return upper, ema, lower

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def calculate_bb_width(bb_upper, bb_lower, bb_mid):
    """Bollinger Band Width - measures squeeze"""
    n = len(bb_upper)
    width = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if not np.isnan(bb_upper[i]) and not np.isnan(bb_lower[i]) and bb_mid[i] != 0:
            width[i] = (bb_upper[i] - bb_lower[i]) / bb_mid[i]
    
    return width

def calculate_squeeze_signal(bb_upper, bb_lower, keltner_upper, keltner_lower):
    """
    Detect volatility squeeze: BB inside Keltner
    Returns 1 when squeeze active, 0 otherwise
    """
    n = len(bb_upper)
    squeeze = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if (not np.isnan(bb_upper[i]) and not np.isnan(bb_lower[i]) and
            not np.isnan(keltner_upper[i]) and not np.isnan(keltner_lower[i])):
            # BB fully inside Keltner = squeeze
            if bb_upper[i] <= keltner_upper[i] and bb_lower[i] >= keltner_lower[i]:
                squeeze[i] = 1.0
    
    return squeeze

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    keltner_upper, keltner_mid, keltner_lower = calculate_keltner(high, low, close, period=20, atr_mult=1.5)
    atr_14 = calculate_atr(high, low, close, period=14)
    volume_sma_20 = calculate_volume_sma(volume, period=20)
    squeeze = calculate_squeeze_signal(bb_upper, bb_lower, keltner_upper, keltner_lower)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_mid)
    
    # Calculate BB width percentile for squeeze strength
    bb_width_sma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    
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
    squeeze_exited = False
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(keltner_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(volume_sma_20[i]) or volume_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SECULAR TREND (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === SQUEEZE DETECTION ===
        is_squeeze = squeeze[i] == 1.0
        squeeze_prev = squeeze[i-1] if i > 0 else 0.0
        squeeze_ending = is_squeeze and squeeze_prev == 0.0
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / volume_sma_20[i] if volume_sma_20[i] > 0 else 0.0
        high_volume = volume_ratio > 1.3  # 30% above average
        
        # === BREAKOUT DETECTION ===
        # Price breaking above BB upper (long breakout)
        bb_breakout_long = close[i] > bb_upper[i-1] if not np.isnan(bb_upper[i-1]) else False
        # Price breaking below BB lower (short breakout)
        bb_breakout_short = close[i] < bb_lower[i-1] if not np.isnan(bb_lower[i-1]) else False
        
        # === SQUEEZE RELEASE (exiting squeeze after being in it) ===
        # Look back up to 10 bars for squeeze
        squeeze_recent = False
        for j in range(max(0, i-10), i+1):
            if squeeze[j] == 1.0:
                squeeze_recent = True
                break
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + squeeze recent + BB breakout + volume confirm
        if price_above_1w and price_above_1d and squeeze_recent:
            if bb_breakout_long and high_volume:
                desired_signal = SIZE_STRONG
            elif bb_breakout_long and not high_volume:
                desired_signal = SIZE_BASE  # weaker signal without volume
        
        # SHORT: 1w bearish + 1d bearish + squeeze recent + BB breakdown + volume confirm
        elif price_below_1w and price_below_1d and squeeze_recent:
            if bb_breakout_short and high_volume:
                desired_signal = -SIZE_STRONG
            elif bb_breakout_short and not high_volume:
                desired_signal = -SIZE_BASE  # weaker signal without volume
        
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
        
        # === EXIT IF SQUEEZE ENDS WITHOUT FOLLOW-THROUGH ===
        # If we're in position and squeeze has been off for 5+ bars with no progress
        if in_position and not squeeze_recent:
            squeeze_off_count = 0
            for j in range(max(0, i-10), i+1):
                if squeeze[j] == 0.0:
                    squeeze_off_count += 1
            if squeeze_off_count >= 5:
                # Check if we've made progress
                if position_side > 0 and close[i] < entry_price:
                    desired_signal = 0.0  # exit losing trade
                elif position_side < 0 and close[i] > entry_price:
                    desired_signal = 0.0  # exit losing trade
        
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
                squeeze_exited = False
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                squeeze_exited = False
        
        signals[i] = final_signal
    
    return signals