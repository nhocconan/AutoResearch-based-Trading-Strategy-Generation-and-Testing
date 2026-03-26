#!/usr/bin/env python3
"""
Experiment #020: 4h Donchian Breakout + Volume Spike + 1d SMA Trend

HYPOTHESIS: Donchian(20) breakout captures momentum after range compression.
Volume spike confirms institutional accumulation/distribution. 1d SMA(50)
filters entries to trade only in the direction of the larger trend.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Breakouts occur in both directions - long in bull breakouts, short in bear
- 1d SMA provides directional bias without being too slow
- Volume spike confirms legitimacy of move
- ATR stoploss adapts to volatility in both markets

TARGET: 100-200 total trades over 4 years (proven pattern from DB).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382)

KEY DESIGN (simple, proven):
1. Donchian(20) breakout above/below
2. Volume > 1.8x 20-avg confirmation
3. 1d SMA(50) for trend direction (only allow entries in trend direction)
4. ATR(14) * 2 for stoploss
5. Signal: 0.25 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_sma50_1d_v1"
timeframe = "4h"
leverage = 1.0

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
    Donchian Channel - uses shift(1) to avoid look-ahead
    Upper = highest high of past period
    Lower = lowest low of past period
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    middle = (upper + lower) / 2.0
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend SMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d SMA(50) for trend direction
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(20) channels
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period for indicators
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
        
        # === 1d TREND DIRECTION ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i] if not np.isnan(sma_1d_aligned[i]) else True
        trend_bullish = price_above_1d_sma
        trend_bearish = not price_above_1d_sma if not np.isnan(sma_1d_aligned[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT LEVELS ===
        upper = donch_upper[i]
        lower = donch_lower[i]
        mid = donch_mid[i]
        
        # Price relative to Donchian
        price_above_upper = close[i] > upper
        price_below_lower = close[i] < lower
        price_near_mid = abs(close[i] - mid) < atr_14[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price breaks above upper band + bullish trend + volume
        if not in_position:
            if price_above_upper and trend_bullish and vol_spike:
                desired_signal = SIZE
        
        # SHORT: Price breaks below lower band + bearish trend + volume
        if not in_position:
            if price_below_lower and trend_bearish and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === OPPOSITE BREAKOUT EXIT ===
        opposite_breakout = False
        if in_position and position_side > 0:
            # Exit long if we get a short breakout signal
            if price_below_lower and trend_bearish:
                opposite_breakout = True
        
        if in_position and position_side < 0:
            # Exit short if we get a long breakout signal
            if price_above_upper and trend_bullish:
                opposite_breakout = True
        
        if opposite_breakout:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals