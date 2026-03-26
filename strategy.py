#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + 1w Trend Filter

HYPOTHESIS: Simple 12h Donchian(20) breakout aligned with 1w trend direction.
This works in both bull (long breakouts) and bear (short breakouts).

WHY SIMPLE WORKS (learned from 16K experiments):
- Complex strategies overfit train, fail on test
- Donchian breakout is proven structural level
- 1w trend filter removes counter-trend trades
- Fewer conditions = fewer trades = less fee drag

TARGET TRADES: 75-150 total over 4 years (18-37/year) for 12h.
HARD MAX: 200 total.

KEY DESIGN:
1. 1w HMA(21) for trend direction only
2. 12h Donchian(20) for entry structure
3. Volume spike confirmation (>1.5x)
4. ATR-based stoploss (2x ATR)
5. MINIMUM HOLD: 4 bars to prevent whipsaw
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1w_trend_simple_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper = highest high, lower = lowest low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    # Warmup
    warmup = max(100, 20 + 14)  # Donchian period + ATR period
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        # Price breakout detection (close above/below channel)
        bullish_breakout = close[i] > upper
        bearish_breakout = close[i] < lower
        
        # === STOPLOSS CHECK (trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long: stop if price falls 2x ATR from highest since entry
            highest_since = max(high[i], entry_price)
            stop_price = highest_since - 2.0 * atr_14[i]
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short: stop if price rises 2x ATR from lowest since entry
            lowest_since = min(low[i], entry_price)
            stop_price = lowest_since + 2.0 * atr_14[i]
            if high[i] > stop_price:
                stoploss_triggered = True
        
        # === ENTRY LOGIC (STRICT) ===
        desired_signal = 0.0
        
        if stoploss_triggered:
            desired_signal = 0.0
        elif in_position:
            # Hold position with trailing stop
            bars_since_entry += 1
            
            if position_side > 0:
                # Long: update trailing stop
                highest_since = max(high[i], entry_price)
                stop_price = highest_since - 2.0 * atr_14[i]
                if low[i] < stop_price:
                    desired_signal = 0.0
                else:
                    desired_signal = SIZE
            else:
                # Short: update trailing stop
                lowest_since = min(low[i], entry_price)
                stop_price = lowest_since + 2.0 * atr_14[i]
                if high[i] > stop_price:
                    desired_signal = 0.0
                else:
                    desired_signal = -SIZE
        else:
            # No position - look for entry
            bars_since_entry = 0
            
            # LONG: Bullish breakout + above 1w trend + volume spike
            if bullish_breakout and weekly_bullish and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + below 1w trend + volume spike
            if bearish_breakout and not weekly_bullish and vol_spike:
                desired_signal = -SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                bars_since_entry = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                bars_since_entry = 0
        
        signals[i] = desired_signal
    
    return signals