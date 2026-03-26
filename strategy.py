#!/usr/bin/env python3
"""
Experiment #029: 4h Donchian Breakout + 1d EMA21 Trend + Volume Spike

HYPOTHESIS: Simple price channel breakout (Donchian) combined with 1d EMA21 trend
direction and volume confirmation captures institutional momentum in both bull
and bear markets. This is the proven pattern from DB that maintains positive Sharpe.

WHY 4h: Optimal trade frequency (20-50/year), reduces fee drag vs lower TFs,
captures multi-day institutional moves vs higher TFs.

WHY NOT COMPLEX: 28 strategies failed with complex conditions. Simple = fewer
false signals = better risk-adjusted returns.

TARGET: 75-200 total trades over 4 years = 19-50/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_ema21_vol_simple_v1"
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
    """Donchian Channels - breakout system"""
    n = len(high)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction (faster than SMA200 = more trades)
    ema_1d_21 = calculate_ema(df_1d['close'].values, 21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_21)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    upper_20, middle_20, lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume - 20 period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 250  # Need 200 for HTF alignment + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(upper_20[i]) or np.isnan(lower_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_ema_1d = close[i] > ema_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        upper = upper_20[i]
        lower = lower_20[i]
        
        # Price breaks above upper channel = bullish breakout
        # Price breaks below lower channel = bearish breakout
        breakout_up = close[i] > upper
        breakout_down = close[i] < lower
        
        # Previous candle was NOT in breakout (avoid re-entry)
        prev_close = close[i - 1] if i > 0 else close[i]
        was_breakout_up = prev_close > upper_20[i - 1] if i > 0 else False
        was_breakout_down = prev_close < lower_20[i - 1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Price breaks above upper channel + above 1d EMA + volume spike
            if breakout_up and not was_breakout_up and price_above_ema_1d and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Price breaks below lower channel + below 1d EMA + volume spike
            if breakout_down and not was_breakout_down and not price_above_ema_1d and vol_spike:
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
        
        # === STOPLOSS EXIT ===
        if in_position:
            if position_side > 0 and low[i] < stop_price:
                desired_signal = 0.0
            if position_side < 0 and high[i] > stop_price:
                desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price drops below 1d EMA
        if in_position and position_side > 0 and close[i] < ema_1d_aligned[i]:
            desired_signal = 0.0
        
        # Exit short if price rises above 1d EMA
        if in_position and position_side < 0 and close[i] > ema_1d_aligned[i]:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
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