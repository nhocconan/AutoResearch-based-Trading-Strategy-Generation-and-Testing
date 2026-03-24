#!/usr/bin/env python3
"""
Experiment #140: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume/Session

Hypothesis: After 122 failed experiments, the pattern is clear:
- Complex regime filters (Choppiness, dual-regime) cause 0 trades or negative Sharpe
- Lower TF (30m/1h) strategies fail due to TOO MANY trades (>200/yr) → fee drag
- BUT #135, #138, #139 had 0 trades → conditions TOO STRICT
- Solution: Use HTF (12h/4h) for DIRECTION, 1h only for ENTRY TIMING
- This gives HTF trade frequency (30-80/yr) with 1h execution precision

Key design choices:
- Timeframe: 1h (as required by experiment)
- HTF: 12h HMA for major trend, 4h RSI for momentum confirmation
- Entry: 1h RSI pullback in trend direction (loose thresholds to ensure trades)
- Volume: >0.8x 20-bar average (confirmation, not hard filter)
- Session: 8-20 UTC bias (higher conviction during liquid hours)
- Position size: 0.25 (conservative for 1h, reduces DD)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
Trade frequency target: 40-80/year (strict enough to avoid fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - more responsive than EMA, less lag
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(series, span):
        """Weighted Moving Average"""
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i-span+1:i+1] * weights) / np.sum(weights)
        return result
    
    close_series = pd.Series(close)
    wma_half = wma(close, period // 2)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    hma = wma(diff, sqrt_period)
    
    return hma

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
    rsi[:] = np.nan
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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000
    hour = pd.to_datetime(ts_seconds, unit='s').hour
    return hour

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
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h RSI for momentum confirmation
    rsi_4h_raw = calculate_rsi(df_4h['close'].values, period=14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi_1h = calculate_rsi(close, period=7)  # Faster RSI for entry timing
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Extract hour for session filter
    hours = np.array([get_hour_from_timestamp(ot) for ot in open_time])
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(rsi_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        # Price above 12h HMA = bullish bias, below = bearish bias
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h MOMENTUM (RSI confirmation) ===
        # Not extreme, just confirmation of direction
        mom_bull = rsi_4h_aligned[i] > 45.0  # Not oversold in uptrend
        mom_bear = rsi_4h_aligned[i] < 55.0  # Not overbought in downtrend
        
        # === 1h ENTRY (RSI pullback) ===
        # Long: RSI pulled back but not extreme (<45)
        # Short: RSI rallied but not extreme (>55)
        entry_long = rsi_1h[i] < 45.0
        entry_short = rsi_1h[i] > 55.0
        
        # === VOLUME CONFIRMATION ===
        # Volume > 0.8x average (not hard filter, just boosts conviction)
        vol_ok = True
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_ratio = volume[i] / vol_sma[i]
            vol_ok = vol_ratio > 0.8
        
        # === SESSION FILTER ===
        # Higher conviction during liquid hours (8-20 UTC)
        session_liquid = 8 <= hours[i] <= 20
        
        # === DESIRED SIGNAL ===
        # LONG: 12h bull + 4h mom bull + 1h entry long + volume ok
        # SHORT: 12h bear + 4h mom bear + 1h entry short + volume ok
        # Session boosts size but doesn't block entry
        
        desired_signal = 0.0
        signal_strength = 1.0
        
        if htf_bull and mom_bull and entry_long and vol_ok:
            desired_signal = SIZE
            if session_liquid:
                signal_strength = 1.2  # Slightly higher conviction in liquid hours
        elif htf_bear and mom_bear and entry_short and vol_ok:
            desired_signal = -SIZE
            if session_liquid:
                signal_strength = 1.2
        
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