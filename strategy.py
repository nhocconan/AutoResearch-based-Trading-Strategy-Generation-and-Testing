#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + 1d Trend + Volume + Chop Filter

HYPOTHESIS: Donchian(20) breakouts are rare events (~1-2 per month per direction).
When combined with 1d HMA trend bias, volume confirmation (1.5x), and choppiness 
filter (CHOP < 55), we get high-quality entries that work in both bull and bear.

WHY THIS WORKS:
- Bull: Long breakouts when 1d HMA bullish (trend continuation)
- Bear: Short breakouts when 1d HMA bearish (trend continuation)
- Volume confirms institutional participation
- Chop filter avoids range-bound whipsaws

TARGET: 75-200 total trades over 4 years (~19-50/year)
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95tr)

KEY FIXES from #017 failure (2443 trades):
1. Donchian breakout (rarer than pivot touches)
2. Stricter volume: 1.5x (not 1.3x)
3. Stricter chop: <55 (not <61.8)
4. Minimum hold: 8 bars (32h) before exit allowed
5. No signal flip - must go flat before reversing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_hma_vol_chop_v3"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    We only enter when CHOP < 55 (trending regime)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper/lower bounds"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA and ratio
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
    bars_in_trade = 0
    MIN_HOLD_BARS = 8  # Must hold at least 8 bars (32h on 4h)
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55  # Stricter: only trending regime
        
        # === TREND BIAS (1d HMA) ===
        hma_1d = hma_1d_aligned[i]
        trend_bullish = (not np.isnan(hma_1d)) and (close[i] > hma_1d)
        trend_bearish = (not np.isnan(hma_1d)) and (close[i] < hma_1d)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # Stricter: 1.5x
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Long breakout: price crosses ABOVE upper Donchian
        # Short breakout: price crosses BELOW lower Donchian
        long_breakout = False
        short_breakout = False
        
        if i > 0 and not np.isnan(donch_upper[i-1]):
            # Price was below upper, now above (or touching)
            if high[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1]:
                long_breakout = True
        
        if i > 0 and not np.isnan(donch_lower[i-1]):
            # Price was above lower, now below (or touching)
            if low[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1]:
                short_breakout = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Donchian breakout + bullish 1d trend + volume spike
            if long_breakout and trend_bullish and vol_spike:
                if not in_position or position_side <= 0:
                    desired_signal = SIZE
            
            # SHORT: Donchian breakout + bearish 1d trend + volume spike
            if short_breakout and trend_bearish and vol_spike:
                if not in_position or position_side >= 0:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === TAKE PROFIT at opposite Donchian ===
        tp_triggered = False
        if in_position and position_side > 0 and not np.isnan(donch_lower[i]):
            if low[i] <= donch_lower[i]:
                tp_triggered = True
        
        if in_position and position_side < 0 and not np.isnan(donch_upper[i]):
            if high[i] >= donch_upper[i]:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLD PERIOD ===
        # Cannot exit before MIN_HOLD_BARS unless stoploss/TP hit
        if in_position and bars_in_trade < MIN_HOLD_BARS:
            if not stoploss_triggered and not tp_triggered:
                # Must maintain position
                desired_signal = signals[i-1] if i > 0 else SIZE * position_side
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            elif np.sign(desired_signal) != position_side:
                # Flip position - only allowed after min hold
                if bars_in_trade >= MIN_HOLD_BARS:
                    position_side = int(np.sign(desired_signal))
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    bars_in_trade = 0
                    if position_side > 0:
                        stop_price = entry_price - 2.5 * entry_atr
                    else:
                        stop_price = entry_price + 2.5 * entry_atr
                # else: keep old position (min hold not met)
        else:
            if in_position:
                if bars_in_trade >= MIN_HOLD_BARS or stoploss_triggered or tp_triggered:
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    stop_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    bars_in_trade = 0
                # else: keep position despite desired_signal=0
        
        if in_position:
            bars_in_trade += 1
        
        signals[i] = desired_signal
    
    return signals