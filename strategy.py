#!/usr/bin/env python3
"""
Experiment #027: 4h Donchian Breakout + Volume + 1d HMA Trend

HYPOTHESIS: Donchian(20) breakouts mark institutional moves. Combined with
volume confirmation (>1.5x) and 1d HMA(21) trend alignment, this captures 
major trends while filtering chop. ATR(14) stoploss at 2.5x limits drawdown.
Simpler = fewer trades = less fee drag = better generalization.

TIMEFRAME: 4h primary
HTF: 1d HMA for trend bias (aligned with shift(1))
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1d_hma_v3"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper = np.full(n, np.nan, dtype=np.float64)
    donch_lower = np.full(n, np.nan, dtype=np.float64)
    donch_mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(19, n):
        donch_upper[i] = np.max(high[i - 20:i + 1])
        donch_lower[i] = np.min(low[i - 20:i + 1])
        donch_mid[i] = (donch_upper[i] + donch_lower[i]) / 2.0
    
    # Previous close to detect crossover
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    
    # Volume MA20
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Need 20 for Donchian + 20 for vol MA + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        # Breakout = close crosses above/below previous channel
        price_above_prev_upper = close[i] > donch_upper[i - 1] if i > 20 else False
        price_below_prev_lower = close[i] < donch_lower[i - 1] if i > 20 else False
        
        # Price already outside current channel
        price_above_channel = close[i] > donch_upper[i]
        price_below_channel = close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above + volume + bullish trend ===
            if price_above_prev_upper or price_above_channel:
                if vol_spike and trend_bullish:
                    desired_signal = SIZE
                # Also allow entry without vol spike but strong trend
                elif trend_bullish and (vol_ratio[i] > 1.2):
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below + volume + bearish trend ===
            if price_below_prev_lower or price_below_channel:
                if vol_spike and trend_bearish:
                    desired_signal = -SIZE
                # Also allow entry without vol spike but strong trend
                elif trend_bearish and (vol_ratio[i] > 1.2):
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit: opposite breakdown OR price returns to mid channel
            if price_below_channel:
                exit_triggered = True
            # Take profit at 3R
            if close[i] > entry_price + 3.0 * entry_atr:
                desired_signal = SIZE / 2  # Half position
        
        if in_position and position_side < 0:
            # Exit: opposite breakout OR price returns to mid channel
            if price_above_channel:
                exit_triggered = True
            # Take profit at 3R
            if close[i] < entry_price - 3.0 * entry_atr:
                desired_signal = -SIZE / 2  # Half position
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_since_entry = 0
            else:
                # Maintain position (could be half sizing from take profit)
                bars_since_entry += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_since_entry = 0
        
        signals[i] = desired_signal
    
    return signals