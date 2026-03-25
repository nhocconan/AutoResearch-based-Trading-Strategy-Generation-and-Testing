#!/usr/bin/env python3
"""
Experiment #1407: 6h Primary + 1d HTF — Regime-Adaptive KAMA + CHOP + RSI

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments per instructions).
This strategy combines regime detection with adaptive entry logic:

1. CHOPPINESS INDEX (14) for regime detection:
   - CHOP > 61.8 = range market → mean reversion logic (RSI extremes)
   - CHOP < 38.2 = trending market → trend-following logic (KAMA slope)
   - This is the "best meta-filter for bear markets" per research notes

2. 1d KAMA(21) for major trend bias (avoid counter-trend trades in bear)

3. 6h KAMA(14) for adaptive trend following (less lag than EMA, less whipsaw than SMA)

4. Asymmetric RSI thresholds based on regime:
   - Range: Long RSI<35, Short RSI>65 (wider bands for mean reversion)
   - Trend: Long RSI>45, Short RSI<55 (looser for trend continuation)

5. ATR(14) trailing stoploss (signal→0 when stopped)

6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should beat current best (KAMA+ROC Sharpe=0.447):
- Regime filter prevents trend-following in choppy markets (major source of losses)
- CHOP works exceptionally well in 2022 crash and 2025 bear/range
- 6h TF = natural 30-50 trades/year (fee-efficient)
- Asymmetric RSI = more trades in trending, fewer in choppy (optimal frequency)

Entry logic (LOOSE to guarantee trades):
- RANGE (CHOP>61.8): Long if RSI<35 + price>1d_KAMA, Short if RSI>65 + price<1d_KAMA
- TREND (CHOP<38.2): Long if KAMA sloping up + RSI>45, Short if KAMA sloping down + RSI<55

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_chop_kama_rsi_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < slow_period + er_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - er_period]):
            signal = abs(close[i] - close[i - er_period])
            noise = 0.0
            for j in range(i - er_period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 0:
                er[i] = signal / noise
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_chop(high, low, close, period=14):
    """Choppiness Index - identifies ranging vs trending markets"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest > lowest and highest > 0:
            tr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
                tr_sum += tr
            
            atr_like = tr_sum / period
            chop[i] = 100 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, er_period=10, fast_period=2, slow_period=14)
    chop_14 = calculate_chop(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_6h[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (CHOP) ===
        chop = chop_14[i]
        is_range = chop > 61.8  # Range market - mean reversion
        is_trend = chop < 38.2  # Trending market - trend follow
        is_neutral = not is_range and not is_trend  # Transition zone
        
        # === TREND BIAS (1d KAMA) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === 6h KAMA SLOPE (trend momentum) ===
        kama_slope = 0.0
        if i >= 3 and not np.isnan(kama_6h[i - 3]):
            kama_slope = kama_6h[i] - kama_6h[i - 3]
        
        kama_rising = kama_slope > 0
        kama_falling = kama_slope < 0
        
        # === RSI ===
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE - LOOSE to guarantee trades) ===
        desired_signal = 0.0
        
        if is_range:
            # RANGE MARKET: Mean reversion at extremes
            # Long: RSI oversold + price above daily trend
            if rsi < 35 and price_above_1d:
                if rsi < 25:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Short: RSI overbought + price below daily trend
            elif rsi > 65 and price_below_1d:
                if rsi > 75:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        elif is_trend:
            # TRENDING MARKET: Trend continuation
            # Long: KAMA rising + RSI bullish + aligned with daily
            if kama_rising and rsi > 45 and price_above_1d:
                if rsi > 55:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Short: KAMA falling + RSI bearish + aligned with daily
            elif kama_falling and rsi < 55 and price_below_1d:
                if rsi < 45:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        else:
            # NEUTRAL/TRANSITION: Reduced position size, wait for clarity
            # Only enter on strong signals
            if rsi < 30 and price_above_1d:
                desired_signal = SIZE_BASE * 0.5
            elif rsi > 70 and price_below_1d:
                desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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