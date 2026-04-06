#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation and 1d EMA trend filter.
# Goes long when price retraces to S3 level (support) with above-average volume and price above 1d EMA200,
# short when price retraces to R3 level (resistance) with volume and price below 1d EMA200.
# Camarilla levels provide statistically significant support/resistance, EMA200 filters trend direction,
// volume confirms retracement validity. Designed for 50-150 total trades over 4 years (12-37/year).

name = "exp_13819_6h_camarilla12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 20
EMA_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close, period):
    """Calculate Camarilla pivot levels"""
    typical_price = (high + low + close) / 3
    pivot = pd.Series(typical_price).rolling(window=period, min_periods=period).mean()
    range_val = pd.Series(high - low).rolling(window=period, min_periods=period).mean()
    
    # Camarilla levels
    r4 = pivot + (range_val * 1.1 / 2)
    r3 = pivot + (range_val * 1.1 / 4)
    r2 = pivot + (range_val * 1.1 / 6)
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    return r3.values, s3.values, pivot.values  # Using R3/S3 for retracement entries

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Camarilla levels and ATR
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    r3_12h, s3_12h, pivot_12h = calculate_camarilla(high_12h, low_12h, close_12h, CAMARILLA_PERIOD)
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # Calculate ATR on 12h data for stop loss
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, ATR_PERIOD)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data for price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation on 6h
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(pivot_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_12h_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Camarilla retracement signals
        # Long when price retraces to S3 support with volume and above EMA
        long_signal = volume_ok and above_ema and close[i] <= s3_12h_aligned[i] and close[i] > pivot_12h_aligned[i]
        # Short when price retraces to R3 resistance with volume and below EMA
        short_signal = volume_ok and below_ema and close[i] >= r3_12h_aligned[i] and close[i] < pivot_12h_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_12h_aligned[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_12h_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below pivot or above R3 (failure)
            if close[i] < pivot_12h_aligned[i] or close[i] > r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above pivot or below S3 (failure)
            if close[i] > pivot_12h_aligned[i] or close[i] < s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals