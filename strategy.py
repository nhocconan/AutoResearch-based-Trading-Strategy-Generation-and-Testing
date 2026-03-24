#!/usr/bin/env python3
"""
Experiment #545: 15m Primary + 4h/1d HTF — RSI Pullback with Session Filter

Hypothesis: 15m timeframe needs SIMPLE entry logic to generate trades (learned from #537, #541 failures).
Previous 15m strategies failed with Sharpe=0.000 due to too many filters = zero trades.

Key changes from failed 15m experiments:
1. SIMPLER entry: HTF trend + RSI(7) pullback only (no CRSI, no Choppiness)
2. RSI(7) instead of RSI(14) - faster response for 15m entries
3. Session filter: 00-12 UTC (London/NY overlap) - reduces noise trades
4. Volume confirmation: >0.8x 20-bar avg (loose filter, not strict)
5. Position size: 0.20 (smaller for 15m frequency)
6. Stoploss: 2.5x ATR with trailing

Strategy logic:
1. 4h HMA(21) = trend bias (long only when price > HMA, short when < HMA)
2. 1d HMA(21) = macro filter (only trade in direction of daily trend)
3. 15m RSI(7) = entry trigger (oversold <35 in uptrend, overbought >65 in downtrend)
4. Session: 00-12 UTC only (crypto London/NY overlap = highest volume)
5. Volume: current > 0.8 * 20-bar avg (confirms interest)
6. ATR(14)*2.5 stoploss on all positions

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 15m
Trade frequency target: 50-100/year (use session + confluence to limit)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_hma_4h1d_session_v2"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index - Wilder's method"""
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
    """Average True Range - Wilder's method"""
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

def calculate_sma(values, period):
    """Simple Moving Average"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    return sma

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array / 1000) % 86400) / 3600
    return hours.astype(int)

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
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma20 = calculate_sma(volume, 20)
    
    # Extract UTC hour for session filter
    utc_hour = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.20
    SIZE_STRONG = 0.25
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER: 00-12 UTC only (London/NY overlap) ===
        in_session = (utc_hour[i] >= 0 and utc_hour[i] <= 12)
        
        # === VOLUME FILTER: >0.8x 20-bar average (loose confirmation) ===
        vol_confirmed = (vol_sma20[i] > 0 and volume[i] > 0.8 * vol_sma20[i])
        
        # === HTF BIAS (4h trend + 1d macro) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === 4h HMA SLOPE (trend confirmation) ===
        hma_4h_slope_bull = False
        hma_4h_slope_bear = False
        if i >= 10 and not np.isnan(hma_4h_aligned[i-10]):
            hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-10]
            hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-10]
        
        # === RSI PULLBACK ENTRY ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        
        # RSI turning up from oversold
        rsi_turning_up = (i > 0 and not np.isnan(rsi_7[i-1]) and rsi_7[i] > rsi_7[i-1])
        rsi_turning_down = (i > 0 and not np.isnan(rsi_7[i-1]) and rsi_7[i] < rsi_7[i-1])
        
        # === ENTRY LOGIC (SIMPLIFIED for trade generation) ===
        desired_signal = 0.0
        
        # LONG entries: HTF bull + RSI pullback + session + volume
        if htf_bull and hma_4h_slope_bull:
            if rsi_extreme_oversold and in_session and vol_confirmed:
                desired_signal = SIZE_STRONG
            elif rsi_oversold and rsi_turning_up and in_session:
                desired_signal = SIZE_ENTRY
            elif rsi_7[i] < 45.0 and rsi_turning_up and in_session and vol_confirmed:
                desired_signal = SIZE_ENTRY * 0.8
        
        # SHORT entries: HTF bear + RSI pullback + session + volume
        elif htf_bear and hma_4h_slope_bear:
            if rsi_extreme_overbought and in_session and vol_confirmed:
                desired_signal = -SIZE_STRONG
            elif rsi_overbought and rsi_turning_down and in_session:
                desired_signal = -SIZE_ENTRY
            elif rsi_7[i] > 55.0 and rsi_turning_down and in_session and vol_confirmed:
                desired_signal = -SIZE_ENTRY * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trailing stop update
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trailing stop update
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_ENTRY * 0.9:
            final_signal = SIZE_ENTRY
        elif desired_signal <= -SIZE_ENTRY * 0.9:
            final_signal = -SIZE_ENTRY
        elif abs(desired_signal) >= SIZE_ENTRY * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_ENTRY * 0.8
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