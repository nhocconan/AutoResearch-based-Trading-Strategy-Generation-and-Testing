#!/usr/bin/env python3
"""
Experiment #1130: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume + Session

Hypothesis: Previous 1h strategies failed due to overly strict entry conditions (0 trades).
This version uses LOOSER but still confluence-based entries to guarantee 40-80 trades/year.

Key innovations:
1. 4h HMA(21) for trend direction (faster than 1d, better for 1h entries)
2. 1h RSI(14) pullback: 35-45 for longs (not 20), 55-65 for shorts (not 80)
3. Volume confirmation: volume > SMA20(volume) * 0.8 (not strict >1.0)
4. Session filter: 08-20 UTC (high liquidity hours)
5. 1d HMA(21) as meta-filter (only trade with daily bias)
6. ATR(14) 2.5x trailing stop
7. Discrete sizing: 0.0, ±0.20, ±0.30

Why this should work:
- 4h HMA is responsive enough for 1h entries but smooth enough to avoid whipsaws
- RSI 35-45/55-65 catches pullbacks without waiting for extreme oversold/overbought
- Volume filter is lenient (0.8x) to not filter out valid entries
- Session filter reduces noise during low-liquidity hours
- Daily HMA ensures we don't trade against major trend

Entry conditions (LOOSE to guarantee trades):
- LONG: 4h_HMA bullish + 1d_HMA bullish + RSI(14) 35-50 + volume > 0.8*SMA20 + session 08-20
- SHORT: 4h_HMA bearish + 1d_HMA bearish + RSI(14) 50-65 + volume > 0.8*SMA20 + session 08-20

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_vol_session_4h1d_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_session_filter(prices):
    """
    Session filter: 08-20 UTC (high liquidity hours)
    Returns boolean array: True if bar is within session
    """
    n = len(prices)
    session = np.zeros(n, dtype=bool)
    
    for i in range(n):
        try:
            # open_time is in milliseconds
            ts = prices['open_time'].iloc[i]
            if isinstance(ts, (int, np.integer)):
                hour = pd.Timestamp(ts, unit='ms').hour
            else:
                hour = pd.Timestamp(ts).hour
            
            # 08-20 UTC (inclusive of 08, exclusive of 21)
            if 8 <= hour < 21:
                session[i] = True
        except:
            session[i] = True  # Default to True if parsing fails
    
    return session

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    
    # Volume SMA(20)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter
    session_active = calculate_session_filter(prices)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma[i]):
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
        
        # === HTF BIAS (4h + 1d HMA alignment) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong alignment: both 4h and 1d agree
        strong_bull = hma_4h_bull and hma_1d_bull
        strong_bear = hma_4h_bear and hma_1d_bear
        
        # === VOLUME FILTER (lenient: >0.8x SMA) ===
        vol_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === SESSION FILTER ===
        session_ok = session_active[i]
        
        # === ENTRY LOGIC (LOOSE to guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 4h+1d bullish + RSI pullback 35-50 + volume + session
        if strong_bull and vol_ok and session_ok:
            if 35.0 <= rsi_14[i] <= 50.0:
                desired_signal = SIZE_BASE
            elif 30.0 <= rsi_14[i] < 35.0:
                desired_signal = SIZE_STRONG
        
        # SHORT: 4h+1d bearish + RSI pullback 50-65 + volume + session
        elif strong_bear and vol_ok and session_ok:
            if 50.0 <= rsi_14[i] <= 65.0:
                desired_signal = -SIZE_BASE
            elif 65.0 < rsi_14[i] <= 70.0:
                desired_signal = -SIZE_STRONG
        
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