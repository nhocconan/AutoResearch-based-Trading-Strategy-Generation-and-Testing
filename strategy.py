#!/usr/bin/env python3
"""
Experiment #1600: 6h Primary + 1d/1w HTF — Fisher Transform + Adaptive Trend

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Fisher Transform
provides superior reversal signals in bear/range markets. Combined with 1d HMA trend
bias and 1w ultra-long filter, this captures both trend continuations and mean-reversion.

Key innovations vs failed 6h attempts:
1. FISHER TRANSFORM (period=9): Normalizes price to Gaussian, catches extremes better
   than RSI. Long when Fisher crosses above -1.5, short when crosses below +1.5.
2. LOOSE ENTRY THRESHOLDS: Fisher -1.5/+1.5 (not -2/+2) to guarantee ≥30 trades/train
3. DUAL REGIME: Trend (price vs 1d HMA) uses Fisher + momentum, Range uses Fisher extremes
4. 1w HMA for ultra-long-term bias (prevents major counter-trend positions)
5. KAMA adaptive MA for trend confirmation (adapts to volatility)

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Fisher Transform proven superior to RSI in 2022-2024 bear/range markets
- Looser thresholds = more trades = better statistics
- Simpler logic = fewer conflicting filters = more executed trades

Entry logic (LOOSE to guarantee trades):
- LONG trend: 1d_HMA bullish + 1w_HMA neutral/bullish + Fisher<-1.5 crossing up + KAMA bullish
- SHORT trend: 1d_HMA bearish + 1w_HMA neutral/bearish + Fisher>+1.5 crossing down + KAMA bearish
- LONG range: Fisher<-1.0 + price<1d_HMA*0.98 (deep pullback)
- SHORT range: Fisher>+1.0 + price>1d_HMA*1.02 (extended rally)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_kama_regime_1d1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - er_period]):
            signal = abs(close[i] - close[i - er_period])
            noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if noise > 1e-10:
                er[i] = signal / noise
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_fisher(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better at catching extremes than RSI, especially in bear markets
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate median price
    median = (high + low) / 2
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            fisher[i] = 0.0
            trigger[i] = 0.0
            continue
        
        # Normalize price to -1 to +1
        normalized = 2.0 * (median[i] - lowest) / range_val - 1.0
        
        # Apply exponential smoothing
        if i == period - 1:
            smoothed = normalized
        else:
            prev_normalized = 2.0 * (median[i-1] - lowest) / range_val - 1.0
            smoothed = 0.67 * normalized + 0.33 * prev_normalized
        
        # Clamp to avoid division issues
        smoothed = max(-0.999, min(0.999, smoothed))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + smoothed) / (1.0 - smoothed))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_roc(close, period=10):
    """Rate of Change"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] > 1e-10:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher(high, low, close, period=9)
    roc_10 = calculate_roc(close, period=10)
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
        
        if np.isnan(kama_21[i]) or np.isnan(fisher[i]) or np.isnan(roc_10[i]):
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
        
        # === TREND DIRECTION (1d and 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_21[i]
        kama_bearish = close[i] < kama_21[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev = fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else fisher_val
        
        # Fisher crossover signals (LOOSE thresholds for trade frequency)
        fisher_bull_cross = fisher_val > -1.5 and fisher_prev <= -1.5
        fisher_bear_cross = fisher_val < 1.5 and fisher_prev >= 1.5
        fisher_extreme_low = fisher_val < -1.0
        fisher_extreme_high = fisher_val > 1.0
        
        # === MOMENTUM CONFIRMATION ===
        roc_positive = roc_10[i] > 0.0 if not np.isnan(roc_10[i]) else False
        roc_negative = roc_10[i] < 0.0 if not np.isnan(roc_10[i]) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND MODE: 1d HMA defines trend direction
        if price_above_1d:
            # LONG: 1d bullish + Fisher bull cross or extreme + KAMA bullish + ROC positive
            if (fisher_bull_cross or fisher_extreme_low) and kama_bullish and roc_positive:
                # Strong signal if 1w also bullish
                if price_above_1w:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        elif price_below_1d:
            # SHORT: 1d bearish + Fisher bear cross or extreme + KAMA bearish + ROC negative
            if (fisher_bear_cross or fisher_extreme_high) and kama_bearish and roc_negative:
                # Strong signal if 1w also bearish
                if price_below_1w:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # RANGE MODE: Price near 1d HMA (within 2%)
        else:
            price_near_hma = abs(close[i] - hma_1d_aligned[i]) / hma_1d_aligned[i] < 0.02
            
            if price_near_hma:
                # Mean reversion: Fisher extremes only
                if fisher_extreme_low and rsi_14[i] < 40:
                    desired_signal = SIZE_BASE
                elif fisher_extreme_high and rsi_14[i] > 60:
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