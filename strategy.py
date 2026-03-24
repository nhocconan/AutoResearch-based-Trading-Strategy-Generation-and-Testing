#!/usr/bin/env python3
"""
Experiment #125: 1h Primary + 4h/1d HTF — Multi-TF Trend Pullback with Volume/Session Filter

Hypothesis: After 100+ failed experiments, the pattern for 1h timeframe is clear:
- 1h strategies fail due to TOO MANY trades (>200/yr) → fee drag kills profit
- Complex regime filters (Choppiness, CRSI) cause 0 trades on 1h
- SUCCESS formula: HTF (4h/1d) for DIRECTION, 1h only for ENTRY TIMING
- Add strict confluence: volume spike + session filter + RSI pullback

This strategy uses 4 confluence filters to ensure few, high-quality trades:
1. 1d HMA(21) = major trend bias (price above/below)
2. 4h HMA(21) = intermediate trend confirmation (aligned with 1d)
3. 1h RSI(14) pullback to 40-60 zone (not extreme, ensures trend continuation)
4. Volume > 1.2x 20-bar average + Session 8-20 UTC (liquidity filter)
5. ATR trailing stoploss (2.5x) for risk management

Key design choices:
- Timeframe: 1h (as required for exp #125)
- HTF: 4h + 1d for trend direction (proven to reduce whipsaws)
- Entry: RSI pullback within trend (not reversal, not breakout)
- Volume filter: ensures institutional participation
- Session filter: 8-20 UTC only (avoid low-liquidity Asian session)
- Position size: 0.25 (25% of capital, conservative for 1h)
- Target trades: 40-80/year (strict filters prevent overtrading)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - smoother and more responsive than EMA
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(series, span):
        """Weighted Moving Average"""
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights, mode='valid')
        return result
    
    close_arr = np.array(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # Need extra bars for WMA calculations
    min_bars = period + sqrt_n
    if n < min_bars:
        return np.full(n, np.nan)
    
    wma_half = wma(close_arr, half)
    wma_full = wma(close_arr, period)
    
    # Align arrays (wma_half is longer than wma_full)
    offset = period - half
    wma_diff = 2 * wma_half[offset:] - wma_full
    
    # Final WMA on the difference
    hma = wma(wma_diff, sqrt_n)
    
    # Pad with NaN to match original length
    result = np.full(n, np.nan)
    start_idx = (period - half) + (sqrt_n - 1) + offset
    result[start_idx:start_idx + len(hma)] = hma
    
    return result

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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    import datetime
    dt = datetime.datetime.utcfromtimestamp(open_time / 1000)
    return dt.hour

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
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
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
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        session_ok = (8 <= utc_hour <= 20)
        
        # === VOLUME FILTER (>1.2x average) ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 0.0
        volume_ok = vol_ratio > 1.2
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        int_bull = close[i] > hma_4h_aligned[i]
        int_bear = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK FILTER (40-60 zone for trend continuation) ===
        # Long: RSI pulled back to 40-55 in uptrend
        # Short: RSI rallied to 45-60 in downtrend
        rsi_pullback_long = 40.0 <= rsi[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi[i] <= 60.0
        
        # === DESIRED SIGNAL (ALL 4 FILTERS MUST ALIGN) ===
        # LONG: 1d bull + 4h bull + RSI pullback + volume + session
        # SHORT: 1d bear + 4h bear + RSI pullback + volume + session
        desired_signal = 0.0
        
        if htf_bull and int_bull and rsi_pullback_long and volume_ok and session_ok:
            desired_signal = SIZE
        elif htf_bear and int_bear and rsi_pullback_short and volume_ok and session_ok:
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