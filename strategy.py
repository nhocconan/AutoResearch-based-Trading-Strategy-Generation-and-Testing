#!/usr/bin/env python3
"""
Experiment #1161: 15m Primary + 1h/4h/1d HTF — Simple Trend Pullback with Session Filter

Hypothesis: After 12 failed 15m/30m/1h experiments (mostly 0 trades), simplicity wins.
Use 4h/1d HMA for trend DIRECTION (HTF), 1h RSI for momentum filter, 15m EMA pullback for ENTRY.
This gives HTF trade frequency (40-100/year) with 15m entry precision.

Key innovations:
1. 4h HMA(21) + 1d HMA(21) alignment for primary trend (call ONCE before loop)
2. 1h RSI(14) for momentum filter — avoid entering at extremes
3. 15m EMA(21) pullback entry — buy dips in uptrend, sell rallies in downtrend
4. Session filter: 00-12 UTC (London/NY overlap = higher volume, cleaner moves)
5. Very LOOSE entry conditions to guarantee trades (learned from 0-trade failures)
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)
7. 2.5x ATR(14) trailing stop for risk management

Why this should work:
- 4h/1d HMA alignment filters out counter-trend trades (major edge)
- 1h RSI 35-65 range avoids overbought/oversold exhaustion
- 15m EMA pullback = better entry price than chasing breakouts
- Session filter reduces noise during low-volume Asian session
- LOOSE conditions guarantee trades (RSI 35-65 not 40-60, price near EMA not exact)

Entry conditions (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 4h_HMA > 1d_HMA + price > 1d_HMA + 1h_RSI 35-65 + 15m price within 2% of EMA21 + 00-12 UTC
- SHORT: 4h_HMA < 1d_HMA + price < 1d_HMA + 1h_RSI 35-65 + 15m price within 2% of EMA21 + 00-12 UTC

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher freq per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_session_4h1d_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    ema_21 = calculate_ema(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(ema_21[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === HTF TREND BIAS ===
        hma_4h_bull = hma_4h_aligned[i] > hma_1d_aligned[i]
        hma_4h_bear = hma_4h_aligned[i] < hma_1d_aligned[i]
        
        price_vs_1d = close[i] / hma_1d_aligned[i]
        price_above_1d = price_vs_1d > 1.0
        price_below_1d = price_vs_1d < 1.0
        
        # === MOMENTUM FILTER (1h RSI) ===
        rsi_1h = rsi_1h_aligned[i]
        rsi_neutral = (rsi_1h >= 35.0 and rsi_1h <= 65.0)
        
        # === PULLBACK ENTRY (15m price near EMA21) ===
        price_vs_ema = close[i] / ema_21[i] if ema_21[i] > 0 else 1.0
        near_ema = (price_vs_ema >= 0.98 and price_vs_ema <= 1.02)  # within 2%
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + price above 1d HMA + RSI neutral + near EMA + in session
        if hma_4h_bull and price_above_1d and rsi_neutral and near_ema:
            if in_session:
                desired_signal = SIZE_BASE
            else:
                # Allow entries outside session but smaller size
                desired_signal = SIZE_BASE * 0.5
        
        # SHORT: HTF bear + price below 1d HMA + RSI neutral + near EMA + in session
        elif hma_4h_bear and price_below_1d and rsi_neutral and near_ema:
            if in_session:
                desired_signal = -SIZE_BASE
            else:
                desired_signal = -SIZE_BASE * 0.5
        
        # Stronger signal if RSI confirms direction
        if desired_signal > 0 and rsi_1h >= 45.0 and rsi_1h <= 60.0:
            desired_signal = SIZE_STRONG
        elif desired_signal < 0 and rsi_1h >= 40.0 and rsi_1h <= 55.0:
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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