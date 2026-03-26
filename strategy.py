#!/usr/bin/env python3
"""
Experiment #013: 4h Donchian Breakout + 12h Trend + Volume

HYPOTHESIS: 4h Donchian(20) breakout with 12h HMA trend alignment and volume confirmation.
Key insight from DB winners: use CLOSE > PRIOR upper band for longs, CLOSE < PRIOR lower for shorts.
12h HMA provides cleaner trend than 1d. Works in both bull (long breakouts with 12h uptrend)
and bear (short breakouts with 12h downtrend).

TIMEFRAME: 4h primary
HTF: 12h for trend direction
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_12h_trend_v1"
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
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i - span + 1:i + 1]).any():
                result[i] = np.sum(series[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range with Wilder smoothing"""
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
    """Donchian Channel - upper and lower bands"""
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
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA for trend direction
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume ratio (current / 20-bar MA)
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
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] >= 1.2
        
        # === DONCHIAN BREAKOUT detection ===
        # KEY: use PRIOR bar's channel boundary (not current) to avoid look-ahead
        prior_upper = donch_upper[i - 1] if i > 0 else np.nan
        prior_lower = donch_lower[i - 1] if i > 0 else np.nan
        
        # Long: close breaks above PRIOR upper band
        breakout_up = not np.isnan(prior_upper) and close[i] > prior_upper
        # Short: close breaks below PRIOR lower band
        breakout_down = not np.isnan(prior_lower) and close[i] < prior_lower
        
        # === STOPLOSS CHECK (2x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            in_position = False
            position_side = 0
            signals[i] = 0.0
            continue
        
        # === EXIT: price back inside channel OR trend reversal ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: trend reversal or price below 12h HMA
            if not price_above_12h_hma:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: trend reversal or price above 12h HMA
            if price_above_12h_hma:
                exit_triggered = True
        
        if exit_triggered:
            in_position = False
            position_side = 0
            signals[i] = 0.0
            continue
        
        # === NEW ENTRY CONDITIONS ===
        if not in_position:
            # LONG: breakout above prior channel + volume + 12h trend up
            if breakout_up and vol_spike and price_above_12h_hma:
                signals[i] = SIZE
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                stop_price = entry_price - 2.0 * entry_atr
            
            # SHORT: breakout below prior channel + volume + 12h trend down
            elif breakout_down and vol_spike and not price_above_12h_hma:
                signals[i] = -SIZE
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                stop_price = entry_price + 2.0 * entry_atr
            
            else:
                signals[i] = 0.0
        
        # === MAINTAIN POSITION ===
        else:
            if position_side > 0:
                signals[i] = SIZE
            else:
                signals[i] = -SIZE
    
    return signals