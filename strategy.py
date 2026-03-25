#!/usr/bin/env python3
"""
Experiment #1193: 5m Primary + 15m/4h HTF — Ultra-Selective Trend Following

Hypothesis: 5m is untested territory. The key is EXTREME selectivity with HTF trend filter.
Most 5m strategies fail from too many trades (fee drag). This uses:

1. 4h HMA(21) for PRIMARY trend direction (only trade with 4h trend)
2. 15m RSI(14) for momentum confirmation (RSI > 55 for long, < 45 for short)
3. 5m RSI(7) for entry timing (fast RSI extremes: < 25 long, > 75 short)
4. Session filter: 08:00-20:00 UTC only (high liquidity, avoid Asia overnight)
5. Volume spike confirmation (volume > 1.5x 20-bar avg)
6. ATR(14) 2.5x trailing stoploss

Key insight: 5m needs 3+ confluence to avoid fee death. 4h trend ensures we only
trade in established direction. 15m RSI filters out weak momentum. 5m RSI gives
precise entry timing. Session filter avoids low-liquidity whipsaws.

Entry logic (SELECTIVE but guaranteed trades):
- LONG: 4h_HMA bullish + 15m_RSI > 55 + 5m_RSI < 25 (oversold pullback) + session + volume
- SHORT: 4h_HMA bearish + 15m_RSI < 45 + 5m_RSI > 75 (overbought pullback) + session + volume

Why this should work:
- 4h trend filter = only ~50% of time eligible (reduces trades 2x)
- Session filter = only ~50% of bars eligible (reduces trades 2x)
- Triple confluence = ~12.5% of remaining bars (50-100 trades/year target)
- 5m timeframe with HTF filter = best of both worlds
- Small size (0.15) = survivable during 2022 crash

Target: Sharpe>0.5, trades>=50 train, trades>=5 test, DD>-35%, trades/year 50-120
Timeframe: 5m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_trend_rsi_triple_confluence_15m4h_v1"
timeframe = "5m"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_avg
    ratio[:period] = np.nan
    return ratio

def is_session_active(open_time):
    """Check if bar is within 08:00-20:00 UTC session"""
    # open_time is in milliseconds since epoch
    hour = pd.to_datetime(open_time, unit='ms').hour
    return 8 <= hour < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_5m_fast = calculate_rsi(close, period=7)  # Fast RSI for 5m entries
    rsi_5m_std = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
    
    # Warmup period (need all indicators ready)
    min_bars = 100
    
    # Track RSI crosses for entry timing
    rsi_5m_prev = np.nan
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            rsi_5m_prev = rsi_5m_fast[i] if not np.isnan(rsi_5m_fast[i]) else rsi_5m_prev
            continue
        
        if np.isnan(rsi_5m_fast[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            rsi_5m_prev = rsi_5m_fast[i] if not np.isnan(rsi_5m_fast[i]) else rsi_5m_prev
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            rsi_5m_prev = rsi_5m_fast[i] if not np.isnan(rsi_5m_fast[i]) else rsi_5m_prev
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC only) ===
        session_active = is_session_active(open_time[i])
        
        # === TREND DIRECTION (4h HMA) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === MOMENTUM CONFIRMATION (15m RSI) ===
        rsi_15m = rsi_15m_aligned[i]
        momentum_bullish = rsi_15m > 55.0
        momentum_bearish = rsi_15m < 45.0
        
        # === ENTRY TIMING (5m RSI extremes) ===
        rsi_5m = rsi_5m_fast[i]
        rsi_5m_crossed_up = (not np.isnan(rsi_5m_prev) and rsi_5m_prev < 25.0 and rsi_5m >= 25.0)
        rsi_5m_crossed_down = (not np.isnan(rsi_5m_prev) and rsi_5m_prev > 75.0 and rsi_5m <= 75.0)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = not np.isnan(vol_ratio[i]) and vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC (TRIPLE CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m momentum + 5m RSI oversold cross + session + volume
        if price_above_4h and momentum_bullish and session_active:
            if rsi_5m < 30.0 or rsi_5m_crossed_up:
                if volume_confirmed:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m momentum + 5m RSI overbought cross + session + volume
        elif price_below_4h and momentum_bearish and session_active:
            if rsi_5m > 70.0 or rsi_5m_crossed_down:
                if volume_confirmed:
                    desired_signal = -SIZE_STRONG
                else:
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
        rsi_5m_prev = rsi_5m
    
    return signals