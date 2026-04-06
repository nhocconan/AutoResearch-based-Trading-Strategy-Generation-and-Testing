#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot with daily trend filter and volume confirmation.
# Camarilla levels provide natural support/resistance zones: S3/R3 for mean reversion, S4/R4 for breakouts.
# In ranging markets (common in 2025), fade at S3/R3 with daily trend filter.
# In trending markets, breakouts at S4/R4 with volume confirmation capture momentum.
# Daily trend filter ensures alignment with higher timeframe momentum.
# Target: 80-150 total trades over 4 years (20-37/year) to balance opportunity and cost.

name = "exp_13271_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC for Camarilla
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
FADE_ZONE = 0.7  # Fade between S3/R3 (70% of range)

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    range_ = high - low
    if range_ == 0:
        return close, close, close, close, close, close
    close_val = close
    s3 = close_val - (range_ * 1.1 / 6)
    s4 = close_val - (range_ * 1.1 / 2)
    r3 = close_val + (range_ * 1.1 / 6)
    r4 = close_val + (range_ * 1.1 / 2)
    pivot = (high + low + close_val) / 3
    return pivot, s3, s4, r3, r4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    pivot, s3, s4, r3, r4 = calculate_camarilla(high_1d, low_1d, close_1d)
    pivot = np.roll(pivot, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    r3 = np.roll(r3, 1)
    r4 = np.roll(r4, 1)
    # First day will have NaN due to roll, which is correct
    
    # Align to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    
    # Calculate daily EMA for trend filter
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d = np.roll(ema_1d, 1)  # Use previous day's EMA
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(pivot_6h[i]) or np.isnan(ema_1d_6h[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using 6-period average for responsiveness)
        if i >= VOLUME_MA_PERIOD:
            volume_ma = np.mean(volume[i-VOLUME_MA_PERIOD:i])
            volume_ok = volume[i] > (volume_ma * VOLUME_THRESHOLD)
        else:
            volume_ok = False
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > ema_1d_6h[i]
        downtrend = close[i] < ema_1d_6h[i]
        
        # Define zones
        # Fade zone: between S3 and R3 (70% of range)
        # Breakout zone: beyond S4/R4
        fade_long_zone = (close[i] > s3_6h[i]) and (close[i] < pivot_6h[i])
        fade_short_zone = (close[i] < r3_6h[i]) and (close[i] > pivot_6h[i])
        breakout_long = close[i] > r4_6h[i]
        breakout_short = close[i] < s4_6h[i]
        
        # Generate signals
        if position == 0:
            # Fade strategy in ranging markets (when trend is weak)
            if abs(close[i] - ema_1d_6h[i]) / ema_1d_6h[i] < 0.02:  # Near daily EMA = ranging
                if fade_long_zone and volume_ok:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif fade_short_zone and volume_ok:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            # Breakout strategy in trending markets
            elif breakout_long and uptrend and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short and downtrend and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals