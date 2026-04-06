#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend strategy using 4h Supertrend for direction and 1d volume filter.
# Uses Supertrend(ATR=10, multiplier=3) on 4h for trend direction to avoid whipsaws.
# Volume > 2.0x MA(20) on 1d confirms institutional participation. Target 80-150 total trades.
# ATR(10) stop at 2.5x with break-even at 1.5R and trail at 3.0R.
# Session filter: 08-20 UTC to avoid low-volume Asian session.

name = "exp_13674_1h_supertrend4h_vol1d"
timeframe = "1h"
leverage = 1.0

# Parameters
SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 10
ATR_STOP_MULTIPLIER = 2.5
ATR_BREAKEVEN = 1.5
ATR_TRAIL_START = 3.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, atr_period, multiplier):
    """Calculate Supertrend indicator"""
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(1, len(close)):
        if np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            supertrend[i] = np.nan
            direction[i] = direction[i-1]
            continue
            
        # Update bands
        if close[i-1] > upper_band[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
            
        if close[i-1] < lower_band[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        # Determine trend
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        # Set Supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
            
    return supertrend, direction

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Supertrend trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Supertrend for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    supertrend_4h, trend_dir_4h = calculate_supertrend(
        high_4h, low_4h, close_4h, SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER
    )
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    trend_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_dir_4h)
    
    # Load 1d data for volume filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume MA for participation filter
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss and trailing
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    max_favorable = 0.0  # track max favorable excursion for trailing
    
    # Start from warmup period
    start = max(SUPERTREND_ATR_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available or outside session
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(trend_dir_4h_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(volume[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Update max favorable
            max_favorable = max(max_favorable, close[i] - entry_price)
            
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                max_favorable = 0.0
                continue
            
            # Break-even stop
            if max_favorable >= (ATR_BREAKEVEN * atr[i]):
                stop_price = max(stop_price, entry_price)
            
            # Trailing stop
            if max_favorable >= (ATR_TRAIL_START * atr[i]):
                trail_price = entry_price + (max_favorable - (ATR_TRAIL_START * atr[i]))
                if close[i] <= trail_price:
                    signals[i] = 0.0
                    position = 0
                    max_favorable = 0.0
                    continue
        
        elif position == -1:  # short position
            # Update max favorable (positive for shorts)
            max_favorable = max(max_favorable, entry_price - close[i])
            
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                max_favorable = 0.0
                continue
            
            # Break-even stop
            if max_favorable >= (ATR_BREAKEVEN * atr[i]):
                stop_price = min(stop_price, entry_price)
            
            # Trailing stop
            if max_favorable >= (ATR_TRAIL_START * atr[i]):
                trail_price = entry_price - (max_favorable - (ATR_TRAIL_START * atr[i]))
                if close[i] >= trail_price:
                    signals[i] = 0.0
                    position = 0
                    max_favorable = 0.0
                    continue
        
        # Volume confirmation from 1d
        volume_ok = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 4h Supertrend
        uptrend = trend_dir_4h_aligned[i] == 1
        downtrend = trend_dir_4h_aligned[i] == -1
        
        # Price relative to Supertrend
        above_st = close[i] > supertrend_4h_aligned[i]
        below_st = close[i] < supertrend_4h_aligned[i]
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend and above_st:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                max_favorable = 0.0
            elif volume_ok and downtrend and below_st:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                max_favorable = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below Supertrend
            if below_st:
                signals[i] = 0.0
                position = 0
                max_favorable = 0.0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above Supertrend
            if above_st:
                signals[i] = 0.0
                position = 0
                max_favorable = 0.0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals