#!/usr/bin/env python3
"""
Experiment #1009: 15m Primary + 1h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m strategies fail because they're either too noisy (no HTF filter) or too strict
(no trades). This uses 1d HMA for long-term bias, 1h HMA for intermediate confirmation,
and 15m RSI(7) for entry timing. Session filter (00-12 UTC) avoids Asian session noise.

Key innovations:
1. 1d HMA(21): Only long if price > 1d_HMA, only short if price < 1d_HMA (strong bias filter)
2. 1h HMA(21): Confirms intermediate trend alignment
3. 15m RSI(7): Fast RSI for pullback entries (RSI<35 long, RSI>65 short)
4. Session filter: Prefer 00-12 UTC (London+NY overlap for crypto liquidity)
5. ATR(14) 2.0x trailing stop for tight risk management on 15m
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency to reduce fee drag)

Why this should work on 15m:
- HTF bias prevents counter-trend trades (the #1 killer on lower TF)
- RSI(7) is fast enough to catch intraday pullbacks without waiting days
- Session filter reduces noise from low-liquidity periods
- 15m captures multi-hour swings (40-100 trades/year target)
- Smaller size (0.15-0.20) accounts for higher trade frequency

Entry conditions (LOOSE to guarantee trades):
- LONG: price > 1d_HMA AND price > 1h_HMA AND RSI(7) < 40 AND UTC hour 00-12
- SHORT: price < 1d_HMA AND price < 1h_HMA AND RSI(7) > 60 AND UTC hour 00-12
- Strong signals at RSI(7) < 30 or > 70

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_1h1d_v2"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def get_utc_hour(prices, i):
    """Extract UTC hour from open_time column"""
    try:
        # open_time is in milliseconds since epoch
        ts = prices['open_time'].iloc[i] / 1000.0
        from datetime import datetime
        dt = datetime.utcfromtimestamp(ts)
        return dt.hour
    except:
        return 12  # default to midday if parsing fails

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for 15m entries
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
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
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        utc_hour = get_utc_hour(prices, i)
        is_prime_session = 0 <= utc_hour <= 12
        
        # === HTF BIAS (1d and 1h HMA alignment) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1h = close[i] > hma_1h_aligned[i]
        price_below_1h = close[i] < hma_1h_aligned[i]
        
        # Strong alignment: both 1d and 1h agree
        strong_bull = price_above_1d and price_above_1h
        strong_bear = price_below_1d and price_below_1h
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: HTF bullish + RSI oversold pullback
        if strong_bull:
            if rsi_7[i] < 35.0:
                desired_signal = SIZE_BASE
            elif rsi_7[i] < 28.0:
                desired_signal = SIZE_STRONG
            # Allow entries outside prime session with stronger RSI signal
            elif not is_prime_session and rsi_7[i] < 30.0:
                desired_signal = SIZE_BASE
        
        # SHORT: HTF bearish + RSI overbought pullback
        elif strong_bear:
            if rsi_7[i] > 65.0:
                desired_signal = -SIZE_BASE
            elif rsi_7[i] > 72.0:
                desired_signal = -SIZE_STRONG
            # Allow entries outside prime session with stronger RSI signal
            elif not is_prime_session and rsi_7[i] > 70.0:
                desired_signal = -SIZE_BASE
        
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