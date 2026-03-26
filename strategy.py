#!/usr/bin/env python3
"""
Experiment #021: 1d Donchian Breakout + Weekly HMA Trend Filter

HYPOTHESIS: Daily Donchian(20) breakouts mark institutional entry points.
Combined with weekly HMA(21) trend direction filter, this catches major trends
while filtering noise. 1d timeframe is slow enough to avoid overtrading but
fast enough to generate meaningful trades. Works in both directions.

TIMEFRAME: 1d primary
HTF: 1w for trend bias
TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_v1"
timeframe = "1d"
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
        weights = np.arange(1, span + 1)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.zeros(n)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
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
    
    # Weekly HMA for trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period (upper and lower)
    donch_upper = np.full(n, np.nan)
    donch_lower = np.full(n, np.nan)
    for i in range(19, n):
        donch_upper[i] = np.max(high[i - 19:i + 1])
        donch_lower[i] = np.min(low[i - 19:i + 1])
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend: bull if price above weekly HMA
        weekly_bull = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        weekly_bear = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_sma[i] if vol_sma[i] > 0 else False
        
        # Donchian breakout detection (price crosses outside channel)
        above_upper = close[i] > donch_upper[i]
        below_lower = close[i] < donch_lower[i]
        
        # Entry logic
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Break above upper band + weekly bull trend + volume
            if above_upper and weekly_bull and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Break below lower band + weekly bear trend + volume
            if below_lower and weekly_bear and vol_spike:
                desired_signal = -SIZE
        
        # Stop loss check
        if in_position and position_side > 0:
            # Long stop: 3 ATR below entry
            if low[i] < entry_price - 3.0 * entry_atr:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Short stop: 3 ATR above entry
            if high[i] > entry_price + 3.0 * entry_atr:
                desired_signal = 0.0
        
        # Exit on opposite signal or channel re-entry
        if in_position and position_side > 0:
            # Exit long if price falls back below upper band
            if below_lower:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if price rises back above lower band
            if above_upper:
                desired_signal = 0.0
        
        # Update position
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
            # Same direction = hold
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
        
        signals[i] = desired_signal
    
    return signals