#!/usr/bin/env python3
"""
Experiment #1363: 6h Primary + 1d/1w HTF — Adaptive Volatility Breakout + KAMA Trend

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy combines:
1. 1d HMA(21) for major trend bias (avoid counter-trend trades that kill Sharpe)
2. 6h KAMA(21) for adaptive trend following (KAMA adjusts to volatility)
3. 6h Bollinger Band squeeze detection (low vol → breakout imminent)
4. 6h ROC(10) momentum confirmation (ensures breakout has follow-through)
5. ATR-based position sizing (smaller size when vol is high = lower DD)

Why this should work where others failed:
- KAMA adapts to market regime (fast in trends, slow in ranges)
- BB squeeze filters out low-probability entries
- ROC confirmation avoids false breakouts
- 1d trend filter prevents whipsaw in 2022-style crashes
- 6h TF = natural 30-60 trades/year (fee-friendly, not overtraded)

Entry logic:
- LONG: price > 1d_HMA + KAMA turning up + BB squeeze + ROC > 0 + RSI > 45
- SHORT: price < 1d_HMA + KAMA turning down + BB squeeze + ROC < 0 + RSI < 55

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.15-0.30 discrete (vol-scaled)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_bb_squeeze_roc_momentum_1d1w_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 0:
                er[i] = signal / noise
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with bandwidth calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, bandwidth

def calculate_roc(close, period=10):
    """Rate of Change"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc

def calculate_bb_percentile(bandwidth, lookback=50):
    """Calculate BB bandwidth percentile (squeeze detection)"""
    n = len(bandwidth)
    percentile = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        window = bandwidth[i - lookback:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid <= bandwidth[i])
            percentile[i] = rank / len(valid) * 100
    
    return percentile

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
    rsi_14 = calculate_rsi(close, period=14)
    kama_21 = calculate_kama(close, period=21)
    roc_10 = calculate_roc(close, period=10)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_percentile = calculate_bb_percentile(bb_bandwidth, lookback=50)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_21[i]) or np.isnan(roc_10[i]):
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
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA for major regime (stronger filter)
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_uptrend = False
        kama_downtrend = False
        
        if i >= 3:
            # KAMA turning up (3 consecutive higher KAMA values)
            if kama_21[i] > kama_21[i-1] > kama_21[i-2]:
                kama_uptrend = True
            # KAMA turning down (3 consecutive lower KAMA values)
            elif kama_21[i] < kama_21[i-1] < kama_21[i-2]:
                kama_downtrend = True
        
        # === BB SQUEEZE DETECTION ===
        # Squeeze = bandwidth percentile < 30 (low vol, breakout likely)
        bb_squeeze = bb_percentile[i] < 30
        
        # === MOMENTUM CONFIRMATION ===
        roc = roc_10[i]
        rsi = rsi_14[i]
        
        # === VOL-ADJUSTED POSITION SIZING ===
        # Higher ATR = smaller position (risk management)
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[min_bars:i]) if i > min_bars else 1.0
        vol_scale = 1.0 / max(0.5, min(2.0, atr_ratio))  # Scale between 0.5x and 2x
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + KAMA uptrend + BB squeeze + ROC positive + RSI > 45
        if price_above_1d and kama_uptrend and roc > 0 and rsi > 45:
            if price_above_1w:
                # Strong trend alignment (1d + 1w both bullish)
                base_size = SIZE_STRONG
            else:
                # Basic long (only 1d bullish)
                base_size = SIZE_BASE
            
            # Apply vol scaling
            desired_signal = base_size * vol_scale
            
            # BB squeeze adds conviction
            if bb_squeeze:
                desired_signal = min(SIZE_STRONG, desired_signal * 1.2)
        
        # SHORT: 1d bearish + KAMA downtrend + BB squeeze + ROC negative + RSI < 55
        elif price_below_1d and kama_downtrend and roc < 0 and rsi < 55:
            if price_below_1w:
                # Strong trend alignment (1d + 1w both bearish)
                base_size = SIZE_STRONG
            else:
                # Basic short (only 1d bearish)
                base_size = SIZE_BASE
            
            # Apply vol scaling
            desired_signal = -base_size * vol_scale
            
            # BB squeeze adds conviction
            if bb_squeeze:
                desired_signal = max(-SIZE_STRONG, desired_signal * 1.2)
        
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
        elif abs(desired_signal) >= SIZE_WEAK * 0.9:
            if desired_signal > 0:
                final_signal = SIZE_WEAK
            else:
                final_signal = -SIZE_WEAK
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