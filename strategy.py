#!/usr/bin/env python3
"""
Experiment #1630: 1h Primary + 4h/1d HTF — Simplified Trend + Mean Reversion

Hypothesis: Previous 1h strategies failed (Sharpe=0.000) due to overly complex entry logic.
This strategy uses SIMPLER conditions that actually trigger trades while maintaining edge.

Key learnings from failures (#1619, #1625, #1629):
- CRSI + Choppiness + Session = TOO RESTRICTIVE (0 trades)
- Multiple confluence filters that must ALL agree = no trades
- Fisher Transform thresholds too narrow

New approach (SIMPLE = TRADES):
1. 4h HMA(21) for trend direction (proven in mtf_hma_rsi_zscore_v1)
2. 1h RSI(14) for entry timing with LOOSE thresholds (35/65 not 30/70)
3. 1h ATR(14) for volatility filter (only trade when ATR > median)
4. Session filter: 08-20 UTC (London/NY overlap = real volume)
5. 1d HMA as meta-filter (only trade in direction of daily trend)

Entry logic (LOOSE to guarantee trades):
- LONG: 4h HMA bullish + 1d HMA bullish + RSI < 55 + ATR above median + session
- SHORT: 4h HMA bearish + 1d HMA bearish + RSI > 45 + ATR above median + session

Why this should work:
- Fewer conditions = more trades (learned from 0-trade failures)
- 4h + 1d alignment = strong trend bias (proven edge)
- RSI 35-65 range = catches pullbacks without waiting for extremes
- Session filter = avoids dead hours (reduces false signals)

Target: Sharpe>0.6, trades≥40/train, trades≥5/test, DD>-35%
Timeframe: 1h
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_simple_4h1d_v1"
timeframe = "1h"
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

def calculate_session_hour(open_time):
    """Extract hour from open_time (Unix timestamp in seconds)"""
    # Binance timestamps are in milliseconds for klines
    if open_time[0] > 1e12:
        ts = open_time / 1000
    else:
        ts = open_time
    
    hours = np.array([(int(t) // 3600) % 24 for t in ts], dtype=np.int32)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate ATR median for volatility filter (rolling 100 bars)
    atr_median = pd.Series(atr_14).rolling(window=100, min_periods=50).median().values
    
    # Session hours
    session_hours = calculate_session_hour(open_time)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(rsi_14[i]):
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
        
        if np.isnan(atr_median[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC = London/NY overlap) ===
        hour = session_hours[i]
        in_session = 8 <= hour <= 20
        
        # === VOLATILITY FILTER (ATR above median) ===
        vol_ok = atr_14[i] > atr_median[i] * 0.8  # relaxed to 0.8x median
        
        # === TREND DIRECTION (4h + 1d HMA alignment) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 4h and 1d must agree for strong signal
        trend_bullish = price_above_4h and price_above_1d
        trend_bearish = price_below_4h and price_below_1d
        
        # === RSI ENTRY (LOOSE thresholds for trades) ===
        rsi_val = rsi_14[i]
        
        # For long: RSI pulled back but not oversold (35-55 range)
        rsi_long_ok = 35 <= rsi_val <= 55
        
        # For short: RSI rallied but not overbought (45-65 range)
        rsi_short_ok = 45 <= rsi_val <= 65
        
        # === ENTRY LOGIC (SIMPLE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h+1d bullish + RSI pullback + session + vol
        if trend_bullish and rsi_long_ok and in_session and vol_ok:
            desired_signal = SIZE_STRONG if rsi_val < 45 else SIZE_BASE
        
        # SHORT: 4h+1d bearish + RSI rally + session + vol
        elif trend_bearish and rsi_short_ok and in_session and vol_ok:
            desired_signal = -SIZE_STRONG if rsi_val > 55 else -SIZE_BASE
        
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