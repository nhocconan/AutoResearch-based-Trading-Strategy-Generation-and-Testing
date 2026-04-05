#!/usr/bin/env python3
"""
Experiment #9955: 6h Camarilla Pivot + Volume Spike + Trend Continuation
Hypothesis: Camarilla pivot levels from daily timeframe provide high-probability reversal/continuation points. 
Breakouts beyond R4/S4 with volume spike continue the trend, while reversals at R3/S3 with volume fade counter-trend.
Works in bull markets (buy R3 bounces, sell R4 breakouts) and bear markets (sell S3 bounces, buy R4 breakdowns).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9955_6h_camarilla_pivot_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
PIVOT_BUFFER = 0.001  # 0.1% buffer to avoid whipsaws

def calculate_pivot_points(high, low, close):
    """Calculate Camarilla pivot levels from previous day's OHLC"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_hl = high - low
    r3 = pivot + (range_hl * 1.1 / 6)
    r4 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 6)
    s4 = pivot - (range_hl * 1.1 / 2)
    return pivot, r3, r4, s3, s4

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
    
    # Load daily data ONCE before loop for pivot points
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    pivot, r3, r4, s3, s4 = calculate_pivot_points(daily_high, daily_low, daily_close)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA for trend filter (20-period)
    ema_trend = calculate_ema(close, 20)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 20) + 1  # EMA and volume MA
    
    for i in range(start, n):
        # Skip if pivot levels not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 20 EMA
        above_ema = close[i] > ema_trend[i]
        below_ema = close[i] < ema_trend[i]
        
        # Price levels with buffer to avoid whipsaws
        r3_level = r3_aligned[i] * (1 + PIVOT_BUFFER)
        r4_level = r4_aligned[i] * (1 + PIVOT_BUFFER)
        s3_level = s3_aligned[i] * (1 - PIVOT_BUFFER)
        s4_level = s4_aligned[i] * (1 - PIVOT_BUFFER)
        
        # Trading logic:
        # 1. Fade at R3/S3 (counter-trend reversal) - only in ranging markets
        # 2. Breakout at R4/S4 (trend continuation) - only with volume spike
        
        # Fade signals at R3/S3 (only when NOT trending strongly)
        fade_r3 = close[i] >= r3_level and close[i] <= r3_level * 1.002 and not above_ema  # Near R3, bearish bias
        fade_s3 = close[i] <= s3_level and close[i] >= s3_level * 0.998 and not below_ema   # Near S3, bullish bias
        
        # Breakout signals at R4/S4 (with volume spike)
        breakout_r4 = close[i] > r4_level and volume_spike
        breakdown_s4 = close[i] < s4_level and volume_spike
        
        # Entry conditions
        long_entry = fade_s3 or breakdown_s4
        short_entry = fade_r3 or breakout_r4
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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
</response>