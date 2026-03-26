#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian Breakout + Volume Spike + 1d SMA Trend Filter

HYPOTHESIS: Donchian(20) breakouts capture institutional moves. Adding volume 
confirmation (1.5x average) reduces false breakouts. 1d SMA(50) as trend filter 
keeps us aligned with the larger trend. Works in both directions.

TIMEFRAME: 4h primary
HTF: 1d for trend bias (SMA50)
TARGET: 75-200 total trades over 4 years (19-50/year)

LESSON FROM FAILURES: #012 had 0 trades because it required:
  - Donchian breakout + volume spike + 1d HMA + 1w HMA (TOO MANY FILTERS)
  
SOLUTION: Only 3 core conditions, no stacking. Simpler = more trades = better stats.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_sma50_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, lower, mid"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

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
    
    # 1d SMA50 for trend bias
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume MA (20-period on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_distance = 0.0
    
    warmup = 100  # Need 50 for SMA50 HTF alignment + 20 for Donchian + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0 or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        bull_trend = close[i] > sma_1d_aligned[i]
        bear_trend = close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Check if price breaks above previous upper band
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Price breaks above Donchian upper + volume spike + bull trend
            if price_above_upper and vol_spike and bull_trend:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Price breaks below Donchian lower + volume spike + bear trend
            if price_below_lower and vol_spike and bear_trend:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long stoploss: price falls 2.5 ATR below entry
            if low[i] < entry_price - 2.5 * entry_atr:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short stoploss: price rises 2.5 ATR above entry
            if high[i] > entry_price + 2.5 * entry_atr:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TRAILING STOP (lock profits) ===
        trailing_stop_triggered = False
        
        if in_position and position_side > 0:
            # Long: trail stop at 2 ATR profit
            max_profit = high[i] - entry_price
            if max_profit > 3.0 * entry_atr:
                # Move stop to breakeven + small buffer
                new_stop = entry_price + 0.5 * entry_atr
                if low[i] < new_stop:
                    trailing_stop_triggered = True
        
        if in_position and position_side < 0:
            # Short: trail stop at 2 ATR profit
            max_profit = entry_price - low[i]
            if max_profit > 3.0 * entry_atr:
                new_stop = entry_price - 0.5 * entry_atr
                if high[i] > new_stop:
                    trailing_stop_triggered = True
        
        if trailing_stop_triggered:
            desired_signal = 0.0
        
        # === OPPOSITE SIGNAL EXIT ===
        if in_position:
            # Long exit: price breaks below lower band
            if position_side > 0 and price_below_lower:
                desired_signal = 0.0
            
            # Short exit: price breaks above upper band
            if position_side < 0 and price_above_upper:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
        
        signals[i] = desired_signal
    
    return signals