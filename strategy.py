#!/usr/bin/env python3
"""
Experiment #1505: 15m Primary + 4h/1d HTF — Session-Filtered Trend Pullback

Hypothesis: 15m timeframe can work IF we use strict HTF filters + session timing.
Most 15m strategies fail due to: (1) too many trades → fee drag, or (2) too strict → 0 trades.

Key components:
1. 1d HMA(21) for major trend bias — only trade WITH daily trend
2. 4h HMA(21) for intermediate trend — must align with 1d
3. 15m RSI(7) for entry timing — faster than RSI14, catches pullbacks
4. Session filter: 00-12 UTC only (London+NY overlap, highest volume)
5. Volume confirmation: taker_buy_volume > 1.3x 20-bar MA
6. ATR expansion: only trade when vol > 1.2x average (avoid dead zones)
7. Stoploss: 2.5x ATR trailing stop
8. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

Why this should work:
- HTF alignment (1d + 4h) prevents counter-trend disasters
- Session filter cuts 50% of bars → fewer trades, higher quality
- RSI(7) is fast enough to catch 15m pullbacks without lag
- Volume filter ensures we trade when institutions are active
- ATR filter avoids choppy low-vol periods

Entry logic (LOOSE enough for trades, strict enough for quality):
- LONG: 1d_HMA bullish + 4h_HMA bullish + RSI(7)<35 + session + volume + ATR
- SHORT: 1d_HMA bearish + 4h_HMA bearish + RSI(7)>65 + session + volume + ATR

Target: Sharpe>0.6, trades>=40/train, trades>=5/test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["taker_buy_volume"].values if "taker_buy_volume" in prices.columns else prices["volume"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR MA for expansion filter
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ma_20[i]) or np.isnan(atr_ma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # open_time is in milliseconds since epoch
        # Convert to hour of day UTC
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 0 <= hour_utc <= 12
        
        # === HTF TREND ALIGNMENT ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Both HTFs must align
        htf_bullish = price_above_4h and price_above_1d
        htf_bearish = price_below_4h and price_below_1d
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.3 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # === ATR EXPANSION FILTER ===
        atr_expansion = atr_14[i] > 1.2 * atr_ma_50[i] if atr_ma_50[i] > 0 else False
        
        # === RSI ENTRY ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        
        # === ENTRY LOGIC (LOOSE enough for trades) ===
        desired_signal = 0.0
        
        # LONG: HTF bullish + RSI oversold + session + volume + ATR
        if htf_bullish and rsi_oversold and in_session and vol_spike and atr_expansion:
            desired_signal = SIZE_STRONG
        elif htf_bullish and rsi_oversold and in_session:
            # Weaker signal without volume/ATR confirmation
            desired_signal = SIZE_BASE
        
        # SHORT: HTF bearish + RSI overbought + session + volume + ATR
        elif htf_bearish and rsi_overbought and in_session and vol_spike and atr_expansion:
            desired_signal = -SIZE_STRONG
        elif htf_bearish and rsi_overbought and in_session:
            # Weaker signal without volume/ATR confirmation
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals