#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversals from S3/R3 with 12h EMA trend filter and volume confirmation.
# Fade at S3/R3 (strong rejection levels) when aligned with higher timeframe trend.
# Volume confirms institutional interest at these key levels.
# Works in both bull/bear: reversals occur in ranging markets and pullbacks in trends.
# Target: 100-200 total trades over 4 years (25-50/year).

name = "exp_13479_6h_camarilla3_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    # Camarilla levels based on previous period's OHLC
    pivot = (high + low + close) / 3
    range_ = high - low
    
    # Key levels: S3, S2, S1, R1, R2, R3, R4
    s3 = close - (range_ * 1.1 / 2)
    s2 = close - (range_ * 1.1 / 4)
    s1 = close - (range_ * 1.1 / 6)
    r1 = close + (range_ * 1.1 / 6)
    r2 = close + (range_ * 1.1 / 4)
    r3 = close + (range_ * 1.1 / 2)
    r4 = close + (range_ * 1.1)
    
    return {
        's3': s3, 's2': s2, 's1': s1,
        'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4
    }

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    # We'll use daily OHLC to calculate Camarilla for the current 6h period
    # Resample to daily using the actual daily data from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from daily data
    camarilla_levels = calculate_camarilla(high_1d, low_1d, close_1d)
    s3_1d = camarilla_levels['s3']
    r3_1d = camarilla_levels['r3']
    
    # Align daily Camarilla levels to 6h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Camarilla reversal signals
        # Long when price touches/bounces from S3 in uptrend
        # Short when price touches/bounces from R3 in downtrend
        # Using high/low for touch, close for confirmation
        touch_s3 = low[i] <= s3_aligned[i]
        touch_r3 = high[i] >= r3_aligned[i]
        
        # Reversal confirmation: price moves back inside the level
        reverse_from_s3 = touch_s3 and (close[i] > s3_aligned[i])
        reverse_from_r3 = touch_r3 and (close[i] < r3_aligned[i])
        
        # Generate signals
        if position == 0:
            # Long setup: bounce from S3 in uptrend with volume
            if reverse_from_s3 and uptrend and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short setup: bounce from R3 in downtrend with volume
            elif reverse_from_r3 and downtrend and volume_ok:
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