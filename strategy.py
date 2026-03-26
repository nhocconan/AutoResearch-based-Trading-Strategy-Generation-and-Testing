#!/usr/bin/env python3
"""
Experiment #021: 1d Donchian(20) Breakout + Volume + 1w Trend

HYPOTHESIS: On 1d timeframe, 20-period Donchian breakouts mark major institutional 
breakout points. Combined with volume confirmation (1.3x MA20) and 1w HMA trend 
alignment, this captures annual/multi-year trend moves. 1d is slowest TF, naturally 
limiting trades to 15-40/year. Works in both bull (long breakouts to new highs) 
and bear (short breakdowns + rallies to HMA resistance).

WHY 1d: 16,000+ experiments show 4h/12h strategies overtrade. 1d naturally 
compresses signal frequency, reducing fee drag. The 1w HMA provides major trend 
bias without whipsawing.

TIMEFRAME: 1d primary
HTF: 1w for trend bias
TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # WMA helper
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
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend bias (bull when price > HMA, bear when price < HMA)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === Calculate local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper = np.full(n, np.nan, dtype=np.float64)
    donch_lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        donch_upper[i] = np.max(high[i - 19:i + 1])
        donch_lower[i] = np.min(low[i - 19:i + 1])
    
    # Previous Donchian for breakout detection
    donch_upper_prev = np.roll(donch_upper, 1)
    donch_lower_prev = np.roll(donch_lower, 1)
    donch_upper_prev[0] = np.nan
    donch_lower_prev[0] = np.nan
    
    # Volume MA20 and ratio
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout up: price crosses above previous upper band
        breakout_up = False
        if i > 0 and not np.isnan(donch_upper_prev[i]):
            breakout_up = (close[i] > donch_upper_prev[i]) and (close[i-1] <= donch_upper_prev[i-1] if i > 1 else True)
        
        # Breakout down: price crosses below previous lower band
        breakout_down = False
        if i > 0 and not np.isnan(donch_lower_prev[i]):
            breakout_down = (close[i] < donch_lower_prev[i]) and (close[i-1] >= donch_lower_prev[i-1] if i > 1 else True)
        
        # Price outside channel
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === NEW LONG ENTRY ===
            # Breakout up OR price above upper channel + volume + bullish 1w trend
            if (breakout_up or price_above_upper) and vol_spike and price_above_1w_hma:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Breakout down OR price below lower channel + volume + bearish 1w trend
            if (breakout_down or price_below_lower) and vol_spike and not price_above_1w_hma:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals