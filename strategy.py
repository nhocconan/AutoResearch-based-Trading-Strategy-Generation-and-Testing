#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + Volume + 1d Trend

HYPOTHESIS: On 12h timeframe, 20-period Donchian channel breaks capture 
institutional moves. Combined with volume spike confirmation (ratio > 1.5) 
and 1d HMA trend alignment, this filters noise while catching major trends.
Simple: entry on channel touch + vol spike + trend match. Works in both 
bull (long breakouts) and bear (short breakdowns).

TIMEFRAME: 12h primary
HTF: 1d for trend bias
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_v3"
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
    """Donchian Channel - returns upper and lower bands"""
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
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position state
    position_side = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    highest = 0.0
    lowest = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            if position_side != 0:
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            if position_side != 0:
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            if position_side != 0:
                position_side = 0
            continue
        
        # === FILTERS ===
        # Trend: price vs 1d HMA
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # Price position relative to Donchian
        above_upper = close[i] > donch_upper[i]
        below_lower = close[i] < donch_lower[i]
        
        desired_signal = 0.0
        
        # === STOPLOSS CHECK ===
        stop_triggered = False
        if position_side != 0:
            if position_side > 0:
                highest = max(highest, high[i])
                stop_price = highest - 2.0 * entry_atr
                if low[i] < stop_price:
                    stop_triggered = True
            else:
                lowest = min(lowest, low[i])
                stop_price = lowest + 2.0 * entry_atr
                if high[i] > stop_price:
                    stop_triggered = True
        
        if stop_triggered:
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            highest = 0.0
            lowest = 0.0
            continue
        
        # === ENTRY LOGIC ===
        if position_side == 0:
            # LONG: price above upper band + vol spike + bullish trend
            if above_upper and vol_spike and price_above_hma:
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest = high[i]
                lowest = low[i]
                desired_signal = SIZE
            
            # SHORT: price below lower band + vol spike + bearish trend
            elif below_lower and vol_spike and price_below_hma:
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest = high[i]
                lowest = low[i]
                desired_signal = -SIZE
        
        # === MAINTAIN POSITION ===
        else:
            desired_signal = SIZE if position_side > 0 else -SIZE
        
        signals[i] = desired_signal
    
    return signals