#!/usr/bin/env python3
"""
Experiment #1641: 15m Primary + 1h/4h/1d HTF — Simple Trend + RSI Pullback

Hypothesis: 15m timeframe has ZERO successful experiments because strategies are TOO COMPLEX.
Simple logic with LOOSE entry conditions will generate trades while HTF filters prevent whipsaw.

Key design choices based on 15m failure analysis (#1629, #1633, #1637 all Sharpe=0.000):
1. NO regime detection (Choppiness kills trades - see #1639 Sharpe=-1.733)
2. NO Fisher Transform extremes (too rare on 15m)
3. Simple 4h HMA for trend bias (stable direction, not whipsaw)
4. RSI(7) for entry (faster than RSI(14), more signals)
5. Session filter: UTC 0-12 only (London+NY overlap = liquidity)
6. ATR volatility filter (avoid dead zones)
7. LOOSE thresholds: RSI <40 long, >60 short (not 30/70)
8. Discrete sizes: 0.15 base, 0.20 strong (smaller for 15m frequency)

Why this beats previous 15m failures:
- Simpler logic = more trades (previous got 0 trades)
- HTF trend filter = fewer false signals than pure 15m
- Session filter = trade during high-volume periods only
- ATR filter = skip dead markets (no movement = no profit)

Target: Sharpe>0.5, trades≥40 train, trades≥5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller than 4h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_4h1d_simple_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

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
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
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
    min_bars = 250  # Need 200 for SMA + buffer
    
    for i in range(min_bars, n):
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
        
        if np.isnan(sma_200[i]):
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
        
        # === SESSION FILTER (UTC 0-12 only) ===
        # Convert open_time to hour (open_time is in milliseconds)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24
        is_session = (hour_utc >= 0 and hour_utc < 12)  # London+NY overlap
        
        # === VOLATILITY FILTER (ATR > threshold) ===
        # Avoid dead markets - ATR should be > 0.5% of price
        atr_pct = atr_14[i] / close[i] * 100
        is_volatile = atr_pct > 0.3  # At least 0.3% ATR
        
        # === TREND DIRECTION (4h HMA bias + 1d confirmation) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Strong trend: 4h and 1d agree
        strong_bull_trend = price_above_4h and price_above_1d
        strong_bear_trend = price_below_4h and price_below_1d
        
        # Weak trend: only 4h signal
        weak_bull_trend = price_above_4h and not price_above_1d
        weak_bear_trend = price_below_4h and not price_below_1d
        
        # === RSI SIGNALS (LOOSE thresholds for trades) ===
        rsi_7_val = rsi_7[i]
        rsi_14_val = rsi_14[i]
        
        # Long signals: RSI oversold
        rsi_7_oversold = rsi_7_val < 40  # LOOSE (not 30)
        rsi_14_oversold = rsi_14_val < 45
        
        # Short signals: RSI overbought
        rsi_7_overbought = rsi_7_val > 60  # LOOSE (not 70)
        rsi_14_overbought = rsi_14_val > 55
        
        # === SMA200 FILTER (only trade in direction of long-term trend) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: Strong trend + RSI pullback + session + volatility
        if strong_bull_trend and rsi_7_oversold and is_session and is_volatile:
            # Bonus if above SMA200
            if price_above_sma200:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # LONG: Weak trend + deep RSI oversold + session + volatility
        elif weak_bull_trend and rsi_7_val < 30 and is_session and is_volatile:
            desired_signal = SIZE_BASE
        
        # SHORT: Strong trend + RSI pullback + session + volatility
        elif strong_bear_trend and rsi_7_overbought and is_session and is_volatile:
            # Bonus if below SMA200
            if price_below_sma200:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # SHORT: Weak trend + deep RSI overbought + session + volatility
        elif weak_bear_trend and rsi_7_val > 70 and is_session and is_volatile:
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