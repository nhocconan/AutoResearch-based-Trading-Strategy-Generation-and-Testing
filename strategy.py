#!/usr/bin/env python3
"""
Experiment #1284: 4h Primary + 12h HTF — Volatility Compression Breakout + RSI Pullback

Hypothesis: Recent failures show two patterns:
- Sharpe=0.000: Entry conditions TOO STRICT (no trades generated)
- Sharpe<0: Entry conditions TOO LOOSE (whipsaw in choppy markets)

This strategy uses:
1. BOLLINGER BAND WIDTH squeeze detection (vol compression = explosive move coming)
2. 12h HMA for macro trend direction (filter false breakouts)
3. RSI(7) pullback entries (enter on retracement, not breakout chase)
4. ATR-based position sizing (reduce size when vol is high)
5. LOOSE thresholds to ensure >=30 trades/symbol/train

Key differences from failed #1281-1283:
- Simpler regime detection (BB width percentile, not Choppiness)
- RSI pullback instead of CRSI extremes (more frequent signals)
- No hysteresis buffer (immediate signal output)
- Lower RSI threshold (35/65 vs 40/60) for more entries

Target: Sharpe > 0.612, trades >= 80 train, >= 12 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_rsi_pullback_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands with bandwidth"""
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    bandwidth = np.full(n, np.nan)
    
    if n < period:
        return mid, upper, lower, bandwidth
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        mid[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
        if mid[i] > 1e-10:
            bandwidth[i] = (upper[i] - lower[i]) / mid[i]
    
    return mid, upper, lower, bandwidth

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_percentile_rank(series, window=100):
    """Percentile rank of current value vs last N values"""
    n = len(series)
    pr = np.full(n, np.nan)
    
    for i in range(window, n):
        window_vals = series[i-window+1:i+1]
        current = series[i]
        rank = np.sum(window_vals[:-1] < current)
        pr[i] = 100.0 * rank / (window - 1)
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    bb_mid, bb_upper, bb_lower, bb_bw = calculate_bollinger(close, period=20, std_mult=2.0)
    bb_bw_percentile = calculate_percentile_rank(bb_bw, window=100)
    rsi = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(bb_bw[i]) or np.isnan(bb_bw_percentile[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(bb_mid[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY REGIME (Bollinger Band Width) ===
        # BB width in bottom 30% = squeeze (explosive move coming)
        bb_squeeze = bb_bw_percentile[i] < 30.0
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 35-45 in bull trend
        rsi_long_pullback = 30.0 < rsi[i] < 50.0
        # Short: RSI rallied to 55-70 in bear trend
        rsi_short_pullback = 50.0 < rsi[i] < 70.0
        
        # === PRICE POSITION IN BB ===
        price_near_lower = close[i] <= bb_lower[i] * 1.005
        price_near_upper = close[i] >= bb_upper[i] * 0.995
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP: Macro bull + RSI pullback + (squeeze OR near BB lower)
        if macro_bull and rsi_long_pullback:
            if bb_squeeze or price_near_lower:
                desired_signal = BASE_SIZE
        
        # SHORT SETUP: Macro bear + RSI pullback + (squeeze OR near BB upper)
        elif macro_bear and rsi_short_pullback:
            if bb_squeeze or price_near_upper:
                desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
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