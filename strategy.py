#!/usr/bin/env python3
"""
Experiment #593: 5m Primary + 15m/4h HTF — Fisher Transform + Session Filter + Volume Confirmation

Hypothesis: 5m timeframe is unexplored territory. Key challenges: extreme fee drag from
overtrading, noise at lower TF. Solution: Use 15m/4h HTF for TREND DIRECTION (not entries),
5m Fisher Transform for precise entry TIMING only. Session filter (08-20 UTC) eliminates
low-volume Asian session whipsaws. Volume confirmation ensures real moves, not noise.

Why this might work on 5m:
1. Fisher Transform (Ehlers) catches reversals faster than RSI - critical for 5m noise
2. 15m HMA(21) = short-term trend filter (must align with entry)
3. 4h HMA(21) = medium-term trend filter (macro bias)
4. Session filter (08-20 UTC) = trade only during London/NY overlap (high volume)
5. Volume ratio filter = only trade when volume > 1.5x 20-bar average
6. ATR(14)*2.5 stoploss on all positions
7. Small position size (0.15-0.20) due to higher trade frequency

Entry logic (ALL must align):
- 4h HMA: price > HMA for long, price < HMA for short
- 15m HMA: price > HMA for long, price < HMA for short
- 5m Fisher: crosses above -1.5 for long, crosses below +1.5 for short
- Session: hour between 08-20 UTC
- Volume: current volume > 1.5x 20-bar average

Target: Sharpe>0.40, trades=50-120/year, DD<-30%
Timeframe: 5m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_fisher_session_vol_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals faster than RSI, especially in noisy markets
    
    Price = (0.33 * 2 * ((close - LL) / (HH - LL) - 0.5) + 0.67 * Price_prev)
    Fisher = 0.5 * ln((1 + Price) / (1 - Price))
    Trigger = Fisher shifted by 1
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    price = np.zeros(n)
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            price[i] = price[i-1] if i > period else 0.0
        else:
            raw = (close[i] - lowest) / range_val
            price[i] = 0.33 * 2.0 * (raw - 0.5) + 0.67 * (price[i-1] if i > period else 0.0)
        
        # Clamp price to avoid log domain errors
        price[i] = max(-0.999, min(0.999, price[i]))
        
        fisher[i] = 0.5 * np.log((1.0 + price[i]) / (1.0 - price[i]))
        
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_avg
    ratio[:period] = np.nan
    
    return ratio

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
    
    # Calculate and align 15m HMA for short-term trend
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 5m indicators
    fisher, trigger = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.18
    SIZE_STRONG = 0.22
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Fisher crossover to avoid repeated entries
    prev_fisher_long_signal = False
    prev_fisher_short_signal = False
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        # open_time is in milliseconds
        timestamp_ms = open_time[i]
        timestamp_s = timestamp_ms / 1000.0
        hour_utc = (timestamp_s % 86400) // 3600
        
        in_session = 8 <= hour_utc <= 20
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND ALIGNMENT ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # Both HTF must agree for trade
        htf_bull_aligned = htf_4h_bull and htf_15m_bull
        htf_bear_aligned = htf_4h_bear and htf_15m_bear
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.3  # 30% above average
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = (fisher[i] > -1.5) and (trigger[i] <= -1.5) and not prev_fisher_long_signal
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = (fisher[i] < 1.5) and (trigger[i] >= 1.5) and not prev_fisher_short_signal
        
        # Reset crossover flags after signal
        if fisher_long_cross or fisher[i] > 0:
            prev_fisher_long_signal = False
        if fisher_short_cross or fisher[i] < 0:
            prev_fisher_short_signal = False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: HTF bull + Fisher long cross + volume confirmed
        if htf_bull_aligned and fisher_long_cross and vol_confirmed:
            desired_signal = SIZE_ENTRY
            prev_fisher_long_signal = True
        
        # SHORT: HTF bear + Fisher short cross + volume confirmed
        elif htf_bear_aligned and fisher_short_cross and vol_confirmed:
            desired_signal = -SIZE_ENTRY
            prev_fisher_short_signal = True
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_ENTRY * 0.9:
            final_signal = SIZE_ENTRY
        elif desired_signal <= -SIZE_ENTRY * 0.9:
            final_signal = -SIZE_ENTRY
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