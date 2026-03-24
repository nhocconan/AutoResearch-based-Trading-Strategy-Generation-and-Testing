#!/usr/bin/env python3
"""
Experiment #020: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume/Session

Hypothesis: After 17 failed experiments, the key insight is that 1h strategies
fail due to OVER-FILTERING (0 trades). This strategy uses PROVEN components
from #011 (4h HMA trend) but adapts for 1h execution with LOOSE filters.

Key design:
1. 4h HMA(21) for trend direction (proven in #011 with Sharpe=0.221)
2. 12h HMA(21) for meta-trend confirmation (adds confluence without over-filtering)
3. 1h RSI(14) for pullback entries (long when RSI<45 in uptrend, short when RSI>55 in downtrend)
4. Volume filter: VERY LOOSE (>0.5x 20-bar avg) - just filters dead periods
5. Session filter: WIDE (5-23 UTC) - captures Asia+Europe+US overlap
6. Size: 0.25 (smaller than 4h strategies due to higher trade frequency)

Why this should work:
- HTF trend filter prevents counter-trend trades (major edge from #011)
- RSI pullback entries catch dips in trends (better than breakouts in chop)
- Loose volume/session filters ensure trades actually happen (avoid 0-trade failure)
- 1h timeframe allows earlier entry than 4h while maintaining HTF direction

Entry Logic:
- Long: 4h close>4h HMA + 12h close>12h HMA + RSI(14)<45 + volume>0.5x avg + hour 5-23
- Short: 4h close<4h HMA + 12h close<12h HMA + RSI(14)>55 + volume>0.5x avg + hour 5-23
- Size: 0.25 (discrete)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.25, trades>40/symbol train, >5/symbol test, DD>-35%
Timeframe: 1h (target 40-80 trades/year with loose filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_loose_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - faster response with less lag than EMA
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n) period
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper function for WMA
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i - span + 1:i + 1] * weights)
        return result
    
    half_period = period // 2
    if half_period < 1:
        half_period = 1
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    raw_hma = 2.0 * wma_half - wma_full
    
    # WMA with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(raw_hma, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum indicator for pullback entries"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
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
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio[:period] = np.nan
    return vol_ratio

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for meta-trend confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Extract UTC hours
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size (smaller for 1h vs 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND ALIGNMENT (4h + 12h) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === VOLUME FILTER (LOOSE - just filter dead periods) ===
        volume_ok = vol_ratio[i] > 0.5  # Very loose threshold
        
        # === SESSION FILTER (WIDE - 5-23 UTC captures major sessions) ===
        hour = hours[i]
        session_ok = (hour >= 5) and (hour <= 23)
        
        # === RSI PULLBACK ENTRY (LOOSE thresholds) ===
        rsi_pullback_long = rsi[i] < 45.0  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 55.0  # Pullback in downtrend
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Dual HMA bullish + RSI pullback + volume + session
        if hma_4h_bull and hma_12h_bull and rsi_pullback_long and volume_ok and session_ok:
            desired_signal = SIZE
        
        # Short entry: Dual HMA bearish + RSI pullback + volume + session
        elif hma_4h_bear and hma_12h_bear and rsi_pullback_short and volume_ok and session_ok:
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
                # Flip position
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