#!/usr/bin/env python3
"""
Experiment #002: Simple 12h Donchian Breakout + 1w Trend

HYPOTHESIS: The simplest possible strategy that works:
- 12h timeframe naturally limits trades to 20-40/year (no overtrading)
- 1w HMA(21) for trend bias ONLY (weekly trend is robust)
- 12h Donchian(20) breakout for entries (proven structure from DB winners)
- Volume confirmation filters noise
- 2.5x ATR stoploss protects against 2022-style crashes

Why it should work in BOTH bull AND bear:
- Bull: Trend-following breakouts in uptrend (HMA up + breakout = continuation)
- Bear: Mean-reversion after bounces (HMA down + breakout short = crash catching)
- Range: Fewer breakouts in range = fewer losses = natural adaptation

Target: 50-150 train trades, Sharpe>0, DD>-40%
Trade frequency: ~20-40/year (12h natural limit)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simple_donchian_1w_trend_v1"
timeframe = "12h"
leverage = 1.0


def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1]
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
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
    
    return pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values


def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout detection"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_volume_confirm(volume, taker_buy_vol, period=20):
    """
    Volume confirmation: compare recent avg to historical avg
    Returns ratio > 1.0 if volume is above average
    """
    n = len(volume)
    if n < period + 1:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = vol_sma > 0
    ratio[mask] = volume[mask] / vol_sma[mask]
    
    return ratio


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_confirm(volume, None, period=20)
    
    signals = np.zeros(n)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    # Warmup
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1w TREND DIRECTION ===
        hma_1w = hma_1w_aligned[i]
        trend_up = close[i] > hma_1w
        trend_down = close[i] < hma_1w
        
        # === DONCHIAN BREAKOUT ===
        # Breakout when price CLOSES above/below prior bar's channel
        prev_idx = i - 1
        if prev_idx >= 0 and not np.isnan(donch_upper[prev_idx]) and not np.isnan(donch_lower[prev_idx]):
            breakout_long = close[i] > donch_upper[prev_idx]
            breakout_short = close[i] < donch_lower[prev_idx]
        else:
            breakout_long = False
            breakout_short = False
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.2x average = confirmation
        vol_confirmed = not np.isnan(vol_ratio[i]) and vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Uptrend (price > 1w HMA) + Donchian breakout + volume confirm
        if trend_up and breakout_long and vol_confirmed:
            desired_signal = 0.30
        
        # SHORT: Downtrend (price < 1w HMA) + Donchian breakdown + volume confirm
        elif trend_down and breakout_short and vol_confirmed:
            desired_signal = -0.30
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === DISCRETIZE ===
        if abs(desired_signal) >= 0.25:
            final_signal = desired_signal
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION ===
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
        
        signals[i] = final_signal
    
    return signals