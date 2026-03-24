#!/usr/bin/env python3
"""
Experiment #896: 30m Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Volume

Hypothesis: 30m timeframe with 4h/1d HTF bias provides optimal trade frequency
(40-80 trades/year) with high signal quality. Ehlers Fisher Transform catches
reversals better than RSI in bear/range markets (proven in literature). 
HMA provides smoother trend bias than EMA. Volume confirmation filters false
breakouts. Session filter (08-20 UTC) captures high liquidity hours.

Key innovations:
1. 1d HMA(21) for primary HTF trend bias
2. 4h HMA(16/48) dual crossover for intermediate trend confirmation
3. Ehlers Fisher Transform (period=9) for reversal detection
4. Volume ratio (vol/MA_vol_20) > 1.2 for confirmation
5. Session filter: 08-20 UTC (high liquidity hours)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.25

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1d HMA bull + 4h HMA bull + Fisher < -1.0 + volume_ratio > 1.0 + session
- SHORT: 1d HMA bear + 4h HMA bear + Fisher > +1.0 + volume_ratio > 1.0 + session

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 30m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_hma_vol_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.zeros(n)
    diff[:] = np.nan
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian normal distribution for better reversal detection
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prev_X
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    x_prev = 0.0
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            # Normalize price within range
            x_raw = (high[i] + low[i]) / 2.0  # Use midprice
            x_norm = 0.67 * ((x_raw - lowest_low) / price_range - 0.5) + 0.67 * x_prev
            
            # Clamp to avoid division by zero
            x_norm = np.clip(x_norm, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + x_norm) / (1.0 - x_norm))
            fisher_signal[i] = x_prev
            
            x_prev = x_norm
        else:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            fisher_signal[i] = x_prev
    
    return fisher, fisher_signal

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
    """Volume ratio: current volume / MA(volume, period)"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    
    # Handle division by zero
    vol_ratio[vol_ratio == np.inf] = np.nan
    vol_ratio[vol_ratio == -np.inf] = np.nan
    
    return vol_ratio

def get_session_hour(open_time):
    """
    Extract UTC hour from open_time (milliseconds timestamp)
    Returns hour 0-23
    """
    # Convert milliseconds to seconds, then to datetime
    timestamps_sec = open_time / 1000.0
    # Extract hour from timestamp (UTC)
    hours = ((timestamps_sec % 86400) / 3600).astype(int)
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    session_hours = get_session_hour(open_time)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        in_session = (session_hours[i] >= 8) and (session_hours[i] <= 20)
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        htf_4h_bull = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        htf_4h_bear = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === FISHER TRANSFORM CONDITIONS ===
        # Fisher < -1.0 = oversold (long opportunity)
        # Fisher > +1.0 = overbought (short opportunity)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # Fisher cross signals (more reliable than absolute levels)
        fisher_cross_long = False
        fisher_cross_short = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Fisher crossing above signal line from below
            fisher_cross_long = (fisher[i-1] <= fisher_signal[i-1]) and (fisher[i] > fisher_signal[i])
            # Fisher crossing below signal line from above
            fisher_cross_short = (fisher[i-1] >= fisher_signal[i-1]) and (fisher[i] < fisher_signal[i])
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.0  # Volume above average
        
        # === ENTRY LOGIC (3+ CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 4h bull + Fisher oversold/cross + volume + session
        if htf_1d_bull and htf_4h_bull and in_session:
            if fisher_oversold and vol_confirmed:
                desired_signal = SIZE_STRONG
            elif fisher_cross_long and vol_confirmed:
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bear + 4h bear + Fisher overbought/cross + volume + session
        elif htf_1d_bear and htf_4h_bear and in_session:
            if fisher_overbought and vol_confirmed:
                desired_signal = -SIZE_STRONG
            elif fisher_cross_short and vol_confirmed:
                desired_signal = -SIZE_BASE
        
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