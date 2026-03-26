#!/usr/bin/env python3
"""
Experiment #021: 12h ATR Channel Breakout + 1d HMA Trend

HYPOTHESIS: 12h ATR channel breakouts mark high-probability reversals
when confirmed by 1d HMA trend alignment. Simple = fewer trades = less fee drag.
12h timeframe reduces overtrading (common 4h failure mode).
- Long: price breaks above 12h upper ATR band + 1d HMA bullish + vol spike
- Short: price breaks below 12h lower ATR band + 1d HMA bearish + vol spike
- Stoploss: 2.5 ATR from entry
- Exit: price returns to channel mid or opposite signal

TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_atr_channel_breakout_1d_hma_v1"
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
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1d volume MA for confirmation
    vol_1d = df_1d['volume'].values
    vol_1d_ma = pd.Series(vol_1d).rolling(window=20, min_periods=10).mean().values
    vol_1d_ratio = vol_1d / np.where(vol_1d_ma > 0, vol_1d_ma, 1)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_ratio)
    
    # Local 12h ATR
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR Channel with lookback=20 (similar to Donchian but ATR-based)
    channel_lookback = 20
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)
    mid_band = np.full(n, 0.0, dtype=np.float64)
    
    for i in range(channel_lookback - 1, n):
        recent_high = np.max(high[i - channel_lookback + 1:i + 1])
        recent_low = np.min(low[i - channel_lookback + 1:i + 1])
        upper_band[i] = recent_high
        lower_band[i] = recent_low
        mid_band[i] = (recent_high + recent_low) / 2.0
    
    # Local volume MA for confirmation
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
        
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
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
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        hma_slope_positive = hma_1d_aligned[i] > hma_1d_aligned[i-1] if i > warmup and not np.isnan(hma_1d_aligned[i-1]) else True
        
        bullish_trend = price_above_1d_hma and hma_slope_positive
        bearish_trend = not price_above_1d_hma and not hma_slope_positive
        
        # === VOLUME CONFIRMATION (local) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CHANNEL BREAKOUT DETECTION ===
        # Breakout up: close above upper band AND was below previous bar
        breakout_up = close[i] > upper_band[i] and close[i-1] <= upper_band[i-1] if i > warmup else False
        # Breakout down: close below lower band AND was above previous bar
        breakout_down = close[i] < lower_band[i] and close[i-1] >= lower_band[i-1] if i > warmup else False
        
        # Price outside channel (continuous)
        price_above_upper = close[i] > upper_band[i]
        price_below_lower = close[i] < lower_band[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price breaks above upper band with bullish trend + volume
            if breakout_up or price_above_upper:
                if bullish_trend and vol_spike:
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price breaks below lower band with bearish trend + volume
            if breakout_down or price_below_lower:
                if bearish_trend and vol_spike:
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
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price falls below lower band OR trend turns bearish
            if price_below_lower:
                exit_triggered = True
            if not price_above_1d_hma:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price rises above upper band OR trend turns bullish
            if price_above_upper:
                exit_triggered = True
            if price_above_1d_hma:
                exit_triggered = True
        
        if exit_triggered:
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
                # Same direction - maintain position
                pass
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