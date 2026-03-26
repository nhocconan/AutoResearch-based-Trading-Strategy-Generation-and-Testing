#!/usr/bin/env python3
"""
Experiment #021: 6h Elder Ray + 1d Trend + 6h Donchian

HYPOTHESIS: Elder Ray (Bull/Bear Power) measures institutional conviction:
- Bull Power > 0 when highs exceed EMA13 = institutions buying
- Bear Power < 0 when lows fall below EMA13 = institutions selling

Combined with:
- 1d EMA21 trend filter (institutions follow weekly trend)
- 6h Donchian channel break (structure breakout)
- Volume confirmation (>1.5x MA)

WHY WORKS IN BULL AND BEAR:
- Bull: Long when Bull Power > 0 + price above 1d EMA21 + break upper Donchian
- Bear: Short when Bear Power < 0 + price below 1d EMA21 + break lower Donchian

TARGET: 50-150 total trades over 4 years (~12-37/year)
SIZE: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1d_donchian_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    return pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, lower, middle"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_atr_percentile(atr, period=14):
    """ATR percentile over lookback - filter choppy periods"""
    n = len(atr)
    if n < period + 50:
        return np.full(n, 0.5)
    
    result = np.full(n, 0.5, dtype=np.float64)
    lookback = 100
    
    for i in range(lookback, n):
        if np.isnan(atr[i]):
            continue
        window = atr[i-lookback+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 50:
            result[i] = (valid < atr[i]).sum() / len(valid)
    
    return result

def calculate_bull_bear_power(high, low, close, ema_period=13):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    n = len(close)
    if n < ema_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    ema = calculate_ema(close, ema_period)
    
    bull_power = high - ema
    bear_power = low - ema
    
    return bull_power, bear_power

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA21 for trend
    ema_1d = calculate_ema(df_1d['close'].values, period=21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_pct = calculate_atr_percentile(atr_14, period=14)
    
    # Donchian channel (6h period)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Elder Ray (institutional power)
    bull_power, bear_power = calculate_bull_bear_power(high, low, close, ema_period=13)
    
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
    
    warmup = 100
    
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
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK: ATR not in bottom 20% (avoid chop) ===
        atr_ok = atr_pct[i] > 0.20
        
        # === WEEKLY TREND (1d EMA21) ===
        weekly_bullish = close[i] > ema_1d_aligned[i]
        weekly_bearish = close[i] < ema_1d_aligned[i]
        
        # === ELDER RAY CONFIRMATION ===
        bull_strong = bull_power[i] > 0 if not np.isnan(bull_power[i]) else False
        bear_strong = bear_power[i] < 0 if not np.isnan(bear_power[i]) else False
        
        # === DONCHIAN BREAKOUT (price at channel edge) ===
        donch_width = donch_upper[i] - donch_lower[i]
        if donch_width > 1e-10:
            upper_touch = close[i] >= donch_upper[i] - 0.5 * atr_14[i]
            lower_touch = close[i] <= donch_lower[i] + 0.5 * atr_14[i]
        else:
            upper_touch = False
            lower_touch = False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC (all conditions must align) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + Bull Power positive + Upper Donchian touch + Volume
        if weekly_bullish and bull_strong and upper_touch and vol_confirm and atr_ok:
            desired_signal = SIZE
        
        # SHORT: Weekly bearish + Bear Power negative + Lower Donchian touch + Volume
        if weekly_bearish and bear_strong and lower_touch and vol_confirm and atr_ok:
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and weekly_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and weekly_bullish:
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