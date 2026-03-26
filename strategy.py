#!/usr/bin/env python3
"""
Experiment #024: 12h Donchian Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: 12h Donchian(20) breakouts capture institutional moves. Using 1d HMA
for trend direction filter eliminates counter-trend trades. Volume confirmation
avoids false breakouts. ATR-based stops provide discipline. 12h timeframe reduces
trade frequency vs 4h, helping generalization to test period.

WHY BOTH BULL AND BEAR: 
- Long: breakout above upper Donchian + price above 1d HMA (confirms uptrend)
- Short: breakout below lower Donchian + price below 1d HMA (confirms downtrend)
- Range: no entries when price near 1d HMA (avoid chop)

TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_vol_atr_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel 20-period
    donch_upper = np.full(n, np.nan, dtype=np.float64)
    donch_mid = np.full(n, np.nan, dtype=np.float64)
    donch_lower = np.full(n, np.nan, dtype=np.float64)
    period_dc = 20
    
    for i in range(period_dc - 1, n):
        window_high = high[i - period_dc + 1:i + 1]
        window_low = low[i - period_dc + 1:i + 1]
        donch_upper[i] = np.max(window_high)
        donch_lower[i] = np.min(window_low)
        donch_mid[i] = (donch_upper[i] + donch_lower[i]) / 2.0
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals array
    signals = np.zeros(n)
    SIZE = 0.30  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Check indicators ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0 or np.isnan(donch_upper[i]):
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
        
        # === TREND DIRECTION (1d HMA) ===
        bullish_trend = close[i] > hma_1d_aligned[i]
        bearish_trend = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN STATE ===
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC (tight: breakout + trend + volume) ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: price breaks above upper band + bullish trend + volume
            if price_above_upper and bullish_trend and vol_spike:
                desired_signal = SIZE
            
            # SHORT: price breaks below lower band + bearish trend + volume
            if price_below_lower and bearish_trend and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
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
        
        # === EXITS ===
        # Exit long: price back below 1d HMA (trend change)
        if in_position and position_side > 0:
            if bearish_trend:
                desired_signal = 0.0
        
        # Exit short: price back above 1d HMA (trend change)
        if in_position and position_side < 0:
            if bullish_trend:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals