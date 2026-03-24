#!/usr/bin/env python3
"""
Experiment #058: 30m Primary + 4h/1d HTF — Volatility Squeeze Breakout with Trend Filter

Hypothesis: Lower TF (30m) can work if we use HTF for DIRECTION and 30m only for ENTRY TIMING.
Key insight from failures: #048 got 0 trades because conditions were too strict.
This strategy uses:
1. 4h HMA for trend direction (only trade WITH HTF trend)
2. 30m Bollinger Band squeeze detection (BB Width < 20th percentile = compression)
3. 30m Volume spike confirmation (volume > 1.0x 20-bar avg, not 1.5x which was too strict)
4. Price breakout above/below BB bands for entry trigger
5. Session filter: 8-20 UTC (highest liquidity, but not too restrictive)
6. ATR 2.5x trailing stoploss

Why this might work:
- BB squeeze precedes volatility expansion (proven pattern)
- HTF trend filter prevents counter-trend trades (major failure mode)
- Volume confirmation filters false breakouts
- Relaxed thresholds ensure 30-80 trades/year (not 0 like #048)
- Discrete signal sizes (0.0, ±0.25) minimize fee churn

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 30m (target 40-80 trades/year with HTF filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_squeeze_vol breakout_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    def wma(data, span):
        res = np.full(len(data), np.nan)
        w = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            res[i] = np.sum(data[i - span + 1:i + 1] * w) / np.sum(w)
        return res
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    double_wma = 2.0 * wma_half - wma_full
    hma = wma(double_wma, sqrt_p)
    return hma

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands with width calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Normalized width
    
    return upper, lower, sma, width

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio[:period] = np.nan
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h/1d HMA for HTF trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger(close, period=20, std_mult=2.0)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    utc_hour = get_utc_hour(open_time)
    
    # Calculate BB Width percentile for squeeze detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=100).apply(
        lambda x: np.sum(x[:-1] < x[-1]) / (len(x) - 1) * 100 if len(x) > 1 else 50.0
    ).values
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size - conservative for 30m
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width[i]):
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
        if np.isnan(vol_ratio[i]) or np.isnan(bb_width_percentile[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h and 1d HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment (at least one HTF agrees)
        htf_bull = hma_4h_bull or hma_1d_bull
        htf_bear = hma_4h_bear or hma_1d_bear
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === BB SQUEEZE DETECTION ===
        bb_squeeze = bb_width_percentile[i] < 25.0  # Width in bottom 25th percentile
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.0  # Above average (relaxed from 1.5x)
        
        # === PRICE BREAKOUT ===
        breakout_long = close[i] > bb_upper[i]
        breakout_short = close[i] < bb_lower[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: HTF bull + squeeze + breakout + volume + session
        if htf_bull and bb_squeeze and breakout_long and vol_confirmed and in_session:
            desired_signal = SIZE
        
        # SHORT: HTF bear + squeeze + breakout + volume + session
        elif htf_bear and bb_squeeze and breakout_short and vol_confirmed and in_session:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals