#!/usr/bin/env python3
"""
Experiment #075: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: Using 4h/1d for trend DIRECTION and 1h only for entry TIMING will
generate fewer but higher-quality trades. Session filter (8-20 UTC) removes
low-volume Asian session whipsaws. Volume confirmation prevents fake breakouts.

Key design choices:
1. 4h HMA(21) for primary trend bias (not 1d - too slow for 1h entries)
2. 1d HMA(21) for secondary confirmation (avoid counter-trend trades)
3. 1h RSI(7) for pullback entries (RSI<45 in uptrend, RSI>55 in downtrend)
4. Session filter: only trade 8-20 UTC (high volume European/US overlap)
5. Volume filter: current volume > 0.8x 20-bar avg (confirm participation)
6. ATR(14)*2.5 trailing stoploss for risk management
7. Position size: 0.25 (conservative for 1h TF)

Target: Sharpe>0.351, trades 30-80/year, DD>-40%
Timeframe: 1h (use HTF for direction, 1h for timing only)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    
    # WMA(period/2)
    wma_half = close_series.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    
    # WMA(period)
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Raw HMA
    raw_hma = 2.0 * wma_half - wma_full
    
    # Smooth with WMA(sqrt(period))
    sqrt_period = int(np.sqrt(period))
    hma = raw_hma.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for primary trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for secondary confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi_1h = calculate_rsi(close, period=7)  # Faster RSI for pullback detection
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_avg_1h = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative position size for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h[i]) or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg_1h[i]) or vol_avg_1h[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_1h[i]
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF CONFIRMATION (1d HMA) ===
        # Only take trades in direction of both 4h and 1d trend
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h PULLBACK DETECTION (RSI) ===
        # In uptrend: wait for RSI pullback to 35-45 zone
        # In downtrend: wait for RSI bounce to 55-65 zone
        rsi_pullback_long = 35.0 <= rsi_1h[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi_1h[i] <= 65.0
        
        # === 1h TREND CONFIRMATION ===
        # Price should be above/below 1h HMA for trend alignment
        price_above_hma = close[i] > hma_1h[i]
        price_below_hma = close[i] < hma_1h[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 1d bull + RSI pullback + volume + session
        if htf_bull and daily_bull and rsi_pullback_long and volume_confirmed and in_session:
            # Additional confirmation: price above 1h HMA (trend resuming)
            if price_above_hma:
                desired_signal = SIZE
        
        # SHORT: 4h bear + 1d bear + RSI bounce + volume + session
        elif htf_bear and daily_bear and rsi_pullback_short and volume_confirmed and in_session:
            # Additional confirmation: price below 1h HMA (trend resuming)
            if price_below_hma:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Position reversal
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals