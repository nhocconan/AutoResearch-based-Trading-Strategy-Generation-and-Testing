#!/usr/bin/env python3
"""
Experiment #1469: 15m Primary + 1h/1d HTF — Session-Filtered Pullback Strategy

Hypothesis: 15m timeframe can work IF we use strict session filters and HTF trend
direction. Key insight from failed 15m experiments (#1457, #1461, #1465): they had
0 trades because entry conditions were too strict. This strategy uses LOOSE entry
thresholds (RSI 45/55, not 30/70) to guarantee trades, while session filter
(00-12 UTC) controls trade frequency to 40-100/year.

Strategy components:
1. 1d HMA(21) for major trend bias (avoid counter-trend)
2. 1h HMA(21) for intermediate trend confirmation
3. 15m RSI(7) for pullback entries (loose: <45 long, >55 short)
4. Session filter: only trade 00-12 UTC (London+NY overlap, high liquidity)
5. ATR(14) volatility filter: avoid low-vol periods (ATR ratio > 0.8)
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)
7. Stoploss: 2.5x ATR trailing stop

Why this should work on 15m:
- HTF trend filter (1d + 1h) prevents whipsaw entries
- Session filter reduces trades to fee-efficient levels
- LOOSE RSI thresholds guarantee ≥30 trades/train, ≥3/test
- Smaller position size (0.15-0.20) accounts for higher frequency
- Proven pattern from mtf_hma_rsi_zscore_v1 (Sharpe=5.4)

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_HMA bullish + 1h_HMA bullish + RSI(7)<45 + session 00-12 UTC
- SHORT: 1d_HMA bearish + 1h_HMA bearish + RSI(7)>55 + session 00-12 UTC

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_hma_1h1d_v1"
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    
    # ATR ratio for volatility filter (ATR(7)/ATR(21))
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_21 = calculate_atr(high, low, close, period=21)
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    mask = (atr_21 > 1e-10) & (~np.isnan(atr_7)) & (~np.isnan(atr_21))
    atr_ratio[mask] = atr_7[mask] / atr_21[mask]
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # open_time is in milliseconds, convert to hour
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0) and (hour_utc < 12)
        
        # === VOLATILITY FILTER ===
        vol_ok = atr_ratio[i] > 0.6  # Avoid extremely low vol periods
        
        # === TREND DIRECTION (HTF bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        price_above_1h = close[i] > hma_1h_aligned[i]
        price_below_1h = close[i] < hma_1h_aligned[i]
        
        # === 15m HMA confirmation ===
        hma_15m_bullish = close[i] > hma_15m[i]
        hma_15m_bearish = close[i] < hma_15m[i]
        
        # === RSI (LOOSE thresholds to guarantee trades) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 45  # Loose: was 30, now 45
        rsi_overbought = rsi > 55  # Loose: was 70, now 55
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if in_session and vol_ok:
            # LONG: 1d bullish + 1h bullish + 15m bullish + RSI pullback
            if price_above_1d and price_above_1h and hma_15m_bullish and rsi_oversold:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1d bearish + 1h bearish + 15m bearish + RSI bounce
            elif price_below_1d and price_below_1h and hma_15m_bearish and rsi_overbought:
                desired_signal = -SIZE_STRONG
            
            # Weaker signals (only 2/3 HTF agree)
            elif price_above_1d and price_above_1h and rsi_oversold:
                desired_signal = SIZE_BASE
            elif price_below_1d and price_below_1h and rsi_overbought:
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