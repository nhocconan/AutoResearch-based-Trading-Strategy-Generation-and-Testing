#!/usr/bin/env python3
"""
Experiment #1480: 1h Primary + 4h/12h HTF — Simplified Trend Pullback with Session Filter

Hypothesis: After 1104 failed strategies, the pattern is clear:
1. Lower TF (1h, 30m) fails with over-filtering → 0 trades (see #1470, #1475, #1478)
2. Higher TF (12h, 1d) works with simpler logic (#1477 Sharpe=0.150, #1472 Sharpe=0.055)
3. Complex regime detection (Choppiness, dual-regime) consistently fails
4. Simple HTF trend + LTF pullback entry is the winning pattern

This strategy uses:
- 12h HMA(21) for macro trend direction (proven in #1477)
- 4h HMA(21) for intermediate trend confirmation
- 1h RSI(14) pullback entries (45-55 range, NOT extremes — ensures trades)
- Volume filter (>0.7x 20-bar avg — loose to allow trades)
- Session filter (8-20 UTC — London/NY overlap for liquidity)
- ATR(14)*2.5 trailing stoploss

Why 1h + 4h/12h should work:
1. 12h HMA prevents trading against macro trend (like #1477)
2. 4h HMA adds intermediate confirmation without over-filtering
3. RSI 45-55 (not 30-70) ensures we get entries during pullbacks
4. Session filter reduces noise during low-liquidity hours
5. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
6. Target: 40-80 trades/year (within 1h limit of 30-60)

Timeframe: 1h
HTF: 4h, 12h (call get_htf_data ONCE before loop!)
Position Size: 0.25 (smaller for lower TF to control DD)
Target: 40-80 trades/year, Sharpe > 0.618, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — faster response than EMA with less lag
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights, mode='valid')
        return np.concatenate([np.full(window - 1, np.nan), result])
    
    close_series = pd.Series(close)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Handle length mismatch
    if len(wma_half) > len(wma_full):
        wma_half = wma_half[:len(wma_full)]
    elif len(wma_full) > len(wma_half):
        wma_full = wma_full[:len(wma_half)]
    
    diff = 2 * wma_half - wma_full
    
    hma = wma(diff, sqrt_period)
    
    # Pad to match original length
    if len(hma) < n:
        pad_len = n - len(hma)
        hma = np.concatenate([np.full(pad_len, np.nan), hma])
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMAs for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h to control DD
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Extract hour for session filter
        hour = get_hour_from_open_time(open_time[i])
        
        # === SESSION FILTER (8-20 UTC — London/NY overlap) ===
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (>0.7x average — loose to allow trades) ===
        volume_ok = volume[i] > 0.7 * vol_sma[i]
        
        # === MACRO TREND (12h HMA) — direction bias ===
        # Only trade in direction of 12h trend
        trend_12h_bull = close[i] > hma_12h_aligned[i]
        trend_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) — confirmation ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (1h HMA) — entry timing ===
        trend_1h_bull = close[i] > hma_1h[i]
        trend_1h_bear = close[i] < hma_1h[i]
        
        # === RSI PULLBACK (45-55 range — NOT extremes, ensures trades) ===
        # Long: RSI pulls back to 45-50 in uptrend
        # Short: RSI rallies to 50-55 in downtrend
        rsi_pullback_long = 45.0 <= rsi[i] <= 52.0
        rsi_pullback_short = 48.0 <= rsi[i] <= 55.0
        
        # === DESIRED SIGNAL — SIMPLIFIED TREND PULLBACK ===
        desired_signal = 0.0
        
        # LONG: 12h bull + 4h bull + 1h pullback + RSI support + session + volume
        if trend_12h_bull and trend_4h_bull:
            if rsi_pullback_long and in_session and volume_ok:
                desired_signal = BASE_SIZE
            elif trend_1h_bull and rsi[i] > 48.0 and in_session:
                desired_signal = BASE_SIZE * 0.7  # Weaker signal
        
        # SHORT: 12h bear + 4h bear + 1h pullback + RSI support + session + volume
        elif trend_12h_bear and trend_4h_bear:
            if rsi_pullback_short and in_session and volume_ok:
                desired_signal = -BASE_SIZE
            elif trend_1h_bear and rsi[i] < 52.0 and in_session:
                desired_signal = -BASE_SIZE * 0.7  # Weaker signal
        
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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