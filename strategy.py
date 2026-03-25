#!/usr/bin/env python3
"""
Experiment #1133: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe is unexplored and requires extreme selectivity. By using
4h HMA for primary trend direction, 15m RSI for momentum confirmation, and 5m RSI
for pullback entry timing, we can capture intraday moves while avoiding counter-trend
whipsaws. Session filter (08-20 UTC) avoids low-liquidity periods that cause slippage.

Key innovations:
1. 4h HMA(21) = primary trend filter (NEVER trade counter-trend on 5m)
2. 15m RSI(14) = momentum confirmation (must align with 4h trend)
3. 5m RSI(7) = entry timing (pullback entries in trend direction)
4. Session filter: 08:00-20:00 UTC only (high liquidity, lower slippage)
5. Small size (0.15-0.20) due to higher trade frequency on 5m
6. ATR(14) 2.0x trailing stop for tight risk management
7. Discrete sizing: 0.0, ±0.15, ±0.20 to minimize fee churn

Why this should work:
- 4h trend filter avoids 2022-style bear market whipsaws
- 15m momentum ensures we're not entering at exhausted moves
- 5m RSI pullback = better entry price than chasing breakouts
- Session filter reduces noise from Asian overnight session
- Small size handles 50-120 trades/year without fee destruction

Entry conditions (designed for adequate trade frequency):
- LONG: 4h_HMA_bull + 15m_RSI>45 + 5m_RSI<35 (pullback in uptrend)
- SHORT: 4h_HMA_bear + 15m_RSI<55 + 5m_RSI>65 (pullback in downtrend)
- Session: hour 08-20 UTC only

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 5m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi_pullback_4h15m_v1"
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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return pd.to_datetime(open_time, unit='ms').hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Close position if outside session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4H TREND FILTER (NEVER trade counter-trend) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15M MOMENTUM CONFIRMATION ===
        rsi_15m_neutral = 40.0 <= rsi_15m_aligned[i] <= 60.0
        rsi_15m_bull = rsi_15m_aligned[i] > 45.0
        rsi_15m_bear = rsi_15m_aligned[i] < 55.0
        
        # === 5M ENTRY TIMING (PULLBACK) ===
        rsi_7_oversold = rsi_7[i] < 35.0
        rsi_7_overbought = rsi_7[i] > 65.0
        rsi_7_extreme_oversold = rsi_7[i] < 25.0
        rsi_7_extreme_overbought = rsi_7[i] > 75.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m momentum ok + 5m pullback
        if hma_4h_bull and rsi_15m_bull:
            if rsi_7_extreme_oversold:
                desired_signal = SIZE_STRONG
            elif rsi_7_oversold:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m momentum ok + 5m pullback
        elif hma_4h_bear and rsi_15m_bear:
            if rsi_7_extreme_overbought:
                desired_signal = -SIZE_STRONG
            elif rsi_7_overbought:
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