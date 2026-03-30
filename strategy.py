#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian Breakout + HTF Trend + Volume

HYPOTHESIS: Donchian(20) breakout is the most reliable price structure signal.
By combining with 1d EMA21 for trend, volume confirmation, and ATR-based exits,
this catches major trend moves in both bull and bear markets.

WHY IT WORKS IN BULL AND BEAR: 
- Bull: Price breaks above 4h Donchian high → long continuation
- Bear: Price breaks below 4h Donchian low → short continuation
- Works in both directions with symmetric logic

TARGET: 100-250 total trades over 4 years (25-62/year). HARD MAX: 400.
Signal size: 0.25-0.30.

KEY INSIGHT: Need ENOUGH trades. Too many strategies fail with <50 trades.
Donchian breakout is loose enough to generate signals but tight enough for edge.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_htf_trend_vol_v1"
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
    """Donchian Channel - use PREVIOUS completed bars only"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    return upper, lower

def calculate_hma(values, period=16):
    """Hull Moving Average"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    # Convert to pandas series
    s = pd.Series(values)
    
    # Calculate weighted moving average
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = s.rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, half_period + 1)) / np.sum(np.arange(1, half_period + 1)), raw=True
    ).values
    wma_full = s.rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)), raw=True
    ).values
    
    hma = 2 * wma_half - wma_full
    hma = pd.Series(hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, sqrt_period + 1)) / np.sum(np.arange(1, sqrt_period + 1)), raw=True
    ).values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d EMA50 for additional confirmation
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (shifted by 1 to avoid look-ahead)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21 and EMA50) ===
        # Bullish: price above both EMAs, EMAs aligned up
        bullish_trend = close[i] > ema_1d_aligned[i] and ema_1d_aligned[i] > ema50_1d_aligned[i]
        # Bearish: price below both EMAs, EMAs aligned down
        bearish_trend = close[i] < ema_1d_aligned[i] and ema_1d_aligned[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT LEVELS ===
        dc_upper = donchian_upper[i]
        dc_lower = donchian_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price breaks above Donchian upper with trend + volume ===
            if bullish_trend and high[i] > dc_upper and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Price breaks below Donchian lower with trend + volume ===
            if bearish_trend and low[i] < dc_lower and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR from entry, trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (2 bars = 8h to avoid churn) ===
        bars_held = i - entry_bar
        
        # === TAKE PROFIT: When price hits opposite Donchian side ===
        if in_position and bars_held >= 2:
            if position_side > 0 and close[i] >= dc_upper:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= dc_lower:
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
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals