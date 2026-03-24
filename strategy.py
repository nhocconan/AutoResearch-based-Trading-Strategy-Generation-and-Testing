#!/usr/bin/env python3
"""
Experiment #1053: 5m Primary + 15m/4h HTF — Session Filter + HMA Trend + RSI Pullback

Hypothesis: 5m timeframe has never been tested. Using strict session filter (08-20 UTC)
combined with 4h HMA trend direction and 15m/5m RSI pullback entries will generate
profitable trades while avoiding noise during low-liquidity hours.

Key innovations:
1. Session filter: Only trade 08:00-20:00 UTC (London/NY overlap = highest liquidity)
2. 4h HMA(21) for primary trend direction (NEVER trade counter-trend on 5m)
3. 15m RSI(14) for intermediate momentum confirmation
4. 5m RSI(7) pullback entries (fast RSI for precise timing)
5. ATR(14) 2.0x trailing stoploss
6. Small position size (0.15) due to higher fee drag on 5m

Why this should work:
- Session filter avoids Asian session chop and weekend gaps
- 4h trend filter prevents counter-trend trades (major failure mode)
- 15m RSI confirms momentum before 5m entry
- 5m RSI pullback catches entries at better prices within trend
- Small size (0.15) accounts for 50-120 trades/year fee drag

Entry conditions (balanced for trades):
- LONG: 4h_HMA bullish + 15m_RSI > 40 + 5m_RSI < 45 (pullback in uptrend)
- SHORT: 4h_HMA bearish + 15m_RSI < 60 + 5m_RSI > 55 (pullback in downtrend)
- Session active: hour 08-20 UTC only

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 5m
Size: 0.15 (discrete: 0.0, ±0.15)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_rsi_pullback_15m4h_v1"
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

def is_session_active(open_times, start_hour=8, end_hour=20):
    """
    Check if bar is within trading session (UTC hours)
    open_times: array of open_time in milliseconds
    Returns: boolean array
    """
    n = len(open_times)
    session_mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        # Convert ms to hours UTC
        hour = (open_times[i] // 3600000) % 24
        if start_hour <= hour < end_hour:
            session_mask[i] = True
    
    return session_mask

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_times = prices["open_time"].values
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
    rsi_5m = calculate_rsi(close, period=7)  # Faster RSI for 5m entries
    
    # Session filter
    session_active = is_session_active(open_times, start_hour=8, end_hour=20)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Small size for 5m due to fee drag
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_5m[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        if not session_active[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND DIRECTION (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === INTERMEDIATE MOMENTUM (15m RSI) ===
        rsi_15m_val = rsi_15m_aligned[i]
        rsi_15m_bull = rsi_15m_val > 40.0  # Not oversold on 15m
        rsi_15m_bear = rsi_15m_val < 60.0  # Not overbought on 15m
        
        # === 5m ENTRY TIMING (RSI Pullback) ===
        rsi_5m_val = rsi_5m[i]
        
        # === ENTRY LOGIC (ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m momentum OK + 5m RSI pullback
        if hma_4h_bull and rsi_15m_bull and rsi_5m_val < 45.0:
            desired_signal = SIZE
        
        # SHORT: 4h bearish + 15m momentum OK + 5m RSI pullback
        elif hma_4h_bear and rsi_15m_bear and rsi_5m_val > 55.0:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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