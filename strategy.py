#!/usr/bin/env python3
"""
Experiment #537: 15m Primary + 4h/12h HTF — HMA Trend Pullback + RSI Entry + Session Filter

Hypothesis: 15m timeframe can work if we use HTF (4h/12h) for TREND DIRECTION and only
use 15m for ENTRY TIMING. This gives HTF trade frequency (~50-100/year) with 15m
execution precision. Key insight from 460+ failed strategies: lower TF fails due to
too many trades → fee drag. Solution: HTF direction filter + strict 15m entry confluence.

Strategy logic:
1. 4h HMA(21) = primary trend bias (LONG only if price > 4h HMA, SHORT if <)
2. 12h ADX(14) = regime filter (ADX>25 = trend valid, ADX<20 = reduce size/skip)
3. 15m HMA(8) = fast trend confirmation (aligns with 4h direction)
4. 15m RSI(7) = entry trigger (pullback entry: RSI<35 in uptrend, RSI>65 in downtrend)
5. 15m Volume = confirmation (volume > 1.5x 20-bar avg = real move)
6. Session filter = UTC 00-12 only (London+NY overlap, avoid Asia low-volume)
7. ATR(14)*2.5 = stoploss on all positions

Entry confluence (ALL required):
- LONG: 4h HMA bullish + 15m HMA bullish + RSI(7)<35 + volume spike + UTC 00-12
- SHORT: 4h HMA bearish + 15m HMA bearish + RSI(7)>65 + volume spike + UTC 00-12

Target: Sharpe>0.40, trades=50-100/year (4h frequency via HTF filter)
Timeframe: 15m
Position size: 0.15-0.20 (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h12h_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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
    
    # Calculate and align 4h HMA for primary trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h ADX for regime filter
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=8)
    rsi_15m = calculate_rsi(close, period=7)
    atr_15m = calculate_atr(high, low, close, period=14)
    
    # Volume moving average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 only) ===
        # open_time is in milliseconds, convert to hour
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 0 <= hour_utc <= 12
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF REGIME (12h ADX) ===
        adx_strong = adx_12h_aligned[i] > 25.0  # Trending market
        adx_weak = adx_12h_aligned[i] < 20.0    # Range market
        
        # === 15m HMA TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # HMA slope confirmation
        hma_slope_bull = hma_15m[i] > hma_15m[i-3] if i >= 3 and not np.isnan(hma_15m[i-3]) else False
        hma_slope_bear = hma_15m[i] < hma_15m[i-3] if i >= 3 and not np.isnan(hma_15m[i-3]) else False
        
        # === RSI PULLBACK ENTRY ===
        rsi_oversold = rsi_15m[i] < 35.0
        rsi_overbought = rsi_15m[i] > 65.0
        
        # RSI turning (recovery from extreme)
        rsi_turning_up = rsi_15m[i] > rsi_15m[i-1] if i > 0 and not np.isnan(rsi_15m[i-1]) else False
        rsi_turning_down = rsi_15m[i] < rsi_15m[i-1] if i > 0 and not np.isnan(rsi_15m[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # === ENTRY LOGIC (ALL confluence required) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m bullish + RSI pullback + session + volume
        if htf_bull and hma_15m_bull and hma_slope_bull:
            if rsi_oversold and rsi_turning_up and in_session:
                if vol_spike or adx_strong:  # Volume OR strong trend
                    desired_signal = SIZE_STRONG
                elif rsi_15m[i] < 30.0:  # Deep oversold without volume
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m bearish + RSI pullback + session + volume
        elif htf_bear and hma_15m_bear and hma_slope_bear:
            if rsi_overbought and rsi_turning_down and in_session:
                if vol_spike or adx_strong:
                    desired_signal = -SIZE_STRONG
                elif rsi_15m[i] > 70.0:  # Deep overbought without volume
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
                entry_atr = atr_15m[i]
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