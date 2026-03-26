#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian Breakout + 1d HMA Trend + Volume

HYPOTHESIS: Clean Donchian(20) breakouts with HTF trend confirmation
is the simplest edge that works across ALL market conditions.
The 20-bar channel captures institutional momentum shifts.

WHY THIS SHOULD WORK:
- Donchian is a price channel breakout — universal across bull/bear/range
- 1d HMA filters entries to trend direction only
- Volume confirms institutional involvement
- ATR stop is tight enough to cut losers fast
- Minimal conditions = fewer trades = less fee drag

TARGET: 75-150 total trades over 4 years (proven sweet spot from DB).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (test Sharpe=1.382)

KEY DESIGN:
1. 4h Donchian(20) breakout as primary signal
2. 1d HMA(21) for trend direction filter
3. Volume spike (>1.5x avg) for confirmation
4. ATR(14) stoploss (2x)
5. No take profit — trail with ATR
6. Discrete signal: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_hma_vol_v1"
timeframe = "4h"
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
    """
    Donchian Channel
    upper = highest(high, period)
    lower = lowest(low, period)
    """
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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    upper_20, lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume moving average (20 bars = ~5 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup - need 20 bars for Donchian + ATR
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(upper_20[i]) or np.isnan(lower_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA) ===
        hma_trend_bullish = close[i] > hma_1d_aligned[i]
        hma_trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        upper = upper_20[i]
        lower = lower_20[i]
        
        # Price broke above 20-bar high
        bullish_breakout = close[i] > upper
        # Price broke below 20-bar low
        bearish_breakout = close[i] < lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        # LONG: Bullish breakout + trend aligned + volume confirm
        long_entry = bullish_breakout and hma_trend_bullish and vol_spike
        
        # SHORT: Bearish breakout + trend aligned + volume confirm
        short_entry = bearish_breakout and hma_trend_bearish and vol_spike
        
        desired_signal = 0.0
        
        if long_entry:
            desired_signal = SIZE
        elif short_entry:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT at opposite Donchian band ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at upper band (conservative: lock in 1.5R profit)
            profit_target = entry_price + 1.5 * entry_atr
            if high[i] >= profit_target:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at lower band
            profit_target = entry_price - 1.5 * entry_atr
            if low[i] <= profit_target:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals