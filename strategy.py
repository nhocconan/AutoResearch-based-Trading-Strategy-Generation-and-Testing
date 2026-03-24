#!/usr/bin/env python3
"""
Experiment #533: 5m Primary + 15m/4h HTF — Session-Filtered Adaptive Trend

Hypothesis: 5m timeframe is unexplored territory. By using 4h HMA for macro trend
direction and 15m RSI for momentum confirmation, we can capture intraday moves
while avoiding whipsaw. Session filter (06-22 UTC) avoids low-liquidity Asian night.
Volume spike confirmation ensures we enter on genuine momentum, not noise.

Key innovations:
1. 4h HMA(21) = macro trend filter (ONLY trade in HTF direction)
2. 15m RSI(7) = momentum confirmation (faster RSI for lower TF)
3. 5m KAMA(10,2,30) = adaptive trend following (fast in trends, slow in chop)
4. Session filter: 06:00-22:00 UTC only (avoid Asian night low liquidity)
5. Volume spike: entry volume > 1.3x 20-bar rolling avg
6. ATR(14)*2.0 stoploss (tighter for 5m timeframe)
7. Size = 0.18 (smaller due to higher trade frequency on 5m)

Why this might work on 5m:
- HTF trend filter prevents counter-trend trades (major cause of 5m failures)
- Session filter avoids choppy low-volume periods
- Volume confirmation filters false breakouts
- KAMA adapts to volatility better than EMA/HMA on lower TF

Target: Sharpe>0.40, trades>=150 train (37/year on 5m), trades>=15 test
Timeframe: 5m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_kama_session_vol_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency - fast in trends, slow in chop
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    hour = (ts_seconds % 86400) / 3600.0
    return int(hour)

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
    
    # Calculate and align 4h HMA for macro trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for momentum confirmation
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=7)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_5m = calculate_rsi(close, period=7)  # Faster RSI for 5m
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
    SIZE_STRONG = 0.22
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi_5m[i]) or np.isnan(vol_ratio[i]):
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
        
        # === SESSION FILTER (06:00-22:00 UTC) ===
        hour = get_session_hour(open_time[i])
        in_session = (hour >= 6) and (hour <= 22)
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h macro trend) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM ===
        rsi_15m_bull = rsi_15m_aligned[i] > 45.0
        rsi_15m_bear = rsi_15m_aligned[i] < 55.0
        rsi_15m_strong_bull = rsi_15m_aligned[i] > 55.0
        rsi_15m_strong_bear = rsi_15m_aligned[i] < 45.0
        
        # === KAMA TREND (5m) ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # KAMA slope (direction) - 10 bar lookback
        kama_slope_bull = kama[i] > kama[i-10] if i >= 10 and not np.isnan(kama[i-10]) else False
        kama_slope_bear = kama[i] < kama[i-10] if i >= 10 and not np.isnan(kama[i-10]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === RSI EXTREMES (5m) ===
        rsi_5m_oversold = rsi_5m[i] < 35.0
        rsi_5m_overbought = rsi_5m[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (4h bullish + 5m confirmation)
        if htf_bull:
            # Strong long: all conditions aligned
            if kama_bull and kama_slope_bull and rsi_15m_strong_bull and vol_spike:
                desired_signal = SIZE_STRONG
            # Standard long: KAMA bull + RSI recovery
            elif kama_bull and rsi_15m_bull and rsi_5m[i] > 40.0 and rsi_5m[i] < 60.0:
                if vol_ratio[i] > 1.1:  # Slight volume confirmation
                    desired_signal = SIZE_BASE
            # Pullback long in strong uptrend
            elif kama_bull and rsi_5m_oversold and rsi_15m_bull:
                desired_signal = SIZE_BASE
        
        # SHORT entries (4h bearish + 5m confirmation)
        elif htf_bear:
            # Strong short: all conditions aligned
            if kama_bear and kama_slope_bear and rsi_15m_strong_bear and vol_spike:
                desired_signal = -SIZE_STRONG
            # Standard short: KAMA bear + RSI rejection
            elif kama_bear and rsi_15m_bear and rsi_5m[i] > 40.0 and rsi_5m[i] < 60.0:
                if vol_ratio[i] > 1.1:
                    desired_signal = -SIZE_BASE
            # Pullback short in strong downtrend
            elif kama_bear and rsi_5m_overbought and rsi_15m_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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