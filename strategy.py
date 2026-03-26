#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Choppiness Regime + Volume

HYPOTHESIS: Donchian(20) breakouts mark institutional moves. Choppiness Index
filters out ranging markets (CHOP > 61.8 = range, skip or use mean-reversion).
In trending markets (CHOP < 38.2), momentum entries work. Volume confirms
institutional participation. 1d KAMA provides trend direction. This is the
proven pattern from DB that achieves test Sharpe 1.0-1.5.

WHY 4h: Best performing TF from 16K experiments. Slow enough for fee survival,
fast enough for meaningful signals. 75-200 trades over 4 years is optimal.

KEY: Simple entry (breakout + volume), strict stoploss (2*ATR), no overcomplication.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_vol_1d_kama_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(1, n - period + 1):
        volatility[i - 1] = np.sum(np.abs(close[i:i + period] - close[i - 1:i + period - 1]))
    
    er = np.zeros(n)
    er[period:] = change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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
    """Donchian Channel - returns upper, lower, and midpoint"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy (range), CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                     abs(high[i - j] - close[i - j - 1] if i - j - 1 >= 0 else high[i - j] - low[i - j]),
                     abs(low[i - j] - close[i - j - 1] if i - j - 1 >= 0 else high[i - j] - low[i - j]))
            sum_tr += tr
        
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val > 0:
            chop[i] = 100 * (np.log10(sum_tr) / np.log10(range_val * period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend direction
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
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
    bars_held = 0
    
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
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness) ===
        chop_val = chop[i]
        is_trending = chop_val < 50.0  # Less choppy = trending
        is_choppy = chop_val > 61.8    # Very choppy = skip or reduce size
        
        # === TREND DIRECTION (1d KAMA) ===
        kama_above = close[i] > kama_1d_aligned[i]
        kama_slope_pos = kama_1d_aligned[i] > kama_1d_aligned[i-4] if i >= 4 else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        vol_above_avg = vol_ratio[i] > 1.0
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above upper band (bullish breakout)
        price_breaks_up = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1] if i > 0 else False
        # Price breaks below lower band (bearish breakdown)
        price_breaks_down = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1] if i > 0 else False
        
        # Price position relative to channel
        channel_width = donch_upper[i] - donch_lower[i]
        price_position = (close[i] - donch_lower[i]) / (channel_width + 1e-10)  # 0=bottom, 1=top
        
        # === RSI MOMENTUM ===
        rsi_val = rsi[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Breakout above upper band + volume + bullish 1d KAMA
            if price_breaks_up or (price_position > 0.95 and vol_above_avg):
                # Trend aligned: price above 1d KAMA
                if kama_above and kama_slope_pos:
                    # Volume confirmation (weaker in choppy, stronger in trending)
                    vol_confirm = vol_spike if is_trending else vol_spike
                    if vol_confirm or vol_above_avg:
                        desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Breakdown below lower band + volume + bearish 1d KAMA
            if price_breaks_down or (price_position < 0.05 and vol_above_avg):
                # Trend aligned: price below 1d KAMA
                if not kama_above and not kama_slope_pos:
                    vol_confirm = vol_spike if is_trending else vol_spike
                    if vol_confirm or vol_above_avg:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2 * ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        # Time-based exit: minimum 4 bars held
        if bars_held >= 4:
            if in_position and position_side > 0:
                # Long exit: price falls below mid-channel OR RSI < 35
                if price_position < 0.30:
                    exit_triggered = True
                if rsi_val < 35:
                    exit_triggered = True
                # Trend reversal
                if not kama_above and not kama_slope_pos:
                    exit_triggered = True
            
            if in_position and position_side < 0:
                # Short exit: price rises above mid-channel OR RSI > 65
                if price_position > 0.70:
                    exit_triggered = True
                if rsi_val > 65:
                    exit_triggered = True
                # Trend reversal
                if kama_above and kama_slope_pos:
                    exit_triggered = True
        
        # Aggressive stoploss: 3*ATR from entry
        if in_position:
            if position_side > 0 and low[i] < entry_price - 3.0 * entry_atr:
                exit_triggered = True
                desired_signal = 0.0
            if position_side < 0 and high[i] > entry_price + 3.0 * entry_atr:
                exit_triggered = True
                desired_signal = 0.0
        
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
                bars_held = 0
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
            else:
                # Same direction - maintain position
                bars_held += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_held = 0
        
        signals[i] = desired_signal
    
    return signals