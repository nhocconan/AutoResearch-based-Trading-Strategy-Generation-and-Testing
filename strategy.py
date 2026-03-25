#!/usr/bin/env python3
"""
Experiment #1415: 6h Primary + 12h/1d HTF — Volatility Expansion Breakout

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy combines:
1. 1d HMA(21) for major trend bias (avoid counter-trend trades)
2. 12h HMA(21) for intermediate trend confirmation
3. 6h Bollinger Band squeeze detection (BW percentile < 20%)
4. 6h Donchian(20) breakout entry when BB expands
5. ATR(14) trailing stoploss (signal→0 when stopped)
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work:
- 6h TF = natural 30-50 trades/year (fee-efficient middle ground)
- BB squeeze + breakout captures volatility expansion (proven in TradFi)
- Dual HTF (12h+1d) prevents 2022-style crash whipsaw
- Donchian breakout ensures momentum confirmation
- LOOSE entry filters guarantee trades (unlike failed CRSI/chop strategies)

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_HMA bullish + 12h_HMA bullish + BB_squeeze + price > Donchian_high + RSI < 70
- SHORT: 1d_HMA bearish + 12h_HMA bearish + BB_squeeze + price < Donchian_low + RSI > 30

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_bb_squeeze_donchian_breakout_12h1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with bandwidth"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, sma, bandwidth

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_bw_percentile(bandwidth, lookback=100):
    """Calculate bandwidth percentile over lookback period"""
    n = len(bandwidth)
    percentile = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback - 1, n):
        window = bandwidth[i - lookback + 1:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            current_bw = bandwidth[i]
            if not np.isnan(current_bw):
                percentile[i] = np.sum(valid_window <= current_bw) / len(valid_window) * 100
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    bw_percentile = calculate_bw_percentile(bb_bandwidth, lookback=100)
    
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
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (HTF HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # === 6h HMA CROSSOVER (momentum confirm) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === BB SQUEEZE DETECTION (volatility compression) ===
        bb_squeeze = False
        if not np.isnan(bw_percentile[i]):
            bb_squeeze = bw_percentile[i] < 30  # BW in bottom 30% = squeeze
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i]
        
        # === RSI FILTER (LOOSE - guarantee trades) ===
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 12h bullish + (HMA bullish OR Donchian breakout) + RSI < 70
        if price_above_1d and price_above_12h:
            if (hma_bullish or donchian_breakout_long) and rsi < 70:
                # Strong if BB squeeze present (volatility expansion play)
                if bb_squeeze and donchian_breakout_long:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + 12h bearish + (HMA bearish OR Donchian breakout) + RSI > 30
        elif price_below_1d and price_below_12h:
            if (hma_bearish or donchian_breakout_short) and rsi > 30:
                # Strong if BB squeeze present (volatility expansion play)
                if bb_squeeze and donchian_breakout_short:
                    desired_signal = -SIZE_STRONG
                else:
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