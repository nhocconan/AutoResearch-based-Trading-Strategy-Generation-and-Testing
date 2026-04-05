#!/usr/bin/env python3
"""
Experiment #7671: 6-hour Camarilla pivot from 1-day with volume confirmation and regime filter.
Hypothesis: Camarilla pivot levels (derived from prior day's range) act as strong support/resistance.
In ranging markets (ADX < 25), fade at R3/S3 levels. In trending markets (ADX >= 25),
breakout through R4/S4 continues the trend. Volume must exceed 1.5x average to confirm.
Targets 100-200 total trades over 4 years (25-50/year) with ADX regime filter reducing whipsaw.
"""

from mtf_data import get_alt_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7671_6h_camarilla_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use prior day's OHLC
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < ADX_PERIOD + 10:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Camarilla levels (based on prior day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    camarilla_height = (high_1d - low_1d) * 1.1
    r4 = close_1d + camarilla_height / 2
    r3 = close_1d + camarilla_height / 4
    s3 = close_1d - camarilla_height / 4
    s4 = close_1d - camarilla_height / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX for regime detection
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (ADX_PERIOD - 1) + tr[i]) / ADX_PERIOD
    
    plus_di = 100 * (np.zeros_like(plus_dm))
    minus_di = 100 * (np.zeros_like(minus_dm))
    
    # Smooth DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    for i in range(len(plus_dm)):
        if i == 0:
            plus_dm_smooth[i] = plus_dm[i]
            minus_dm_smooth[i] = minus_dm[i]
        else:
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (ADX_PERIOD - 1) + plus_dm[i]) / ADX_PERIOD
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (ADX_PERIOD - 1) + minus_dm[i]) / ADX_PERIOD
    
    # Avoid division by zero
    dx = np.zeros_like(atr)
    di_sum = plus_dm_smooth + minus_dm_smooth
    mask = di_sum > 0
    dx[mask] = 100 * np.abs(plus_dm_smooth[mask] - minus_dm_smooth[mask]) / di_sum[mask]
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < ADX_PERIOD:
            adx[i] = 0
        elif i == ADX_PERIOD:
            adx[i] = np.mean(dx[1:ADX_PERIOD+1])
        else:
            adx[i] = (adx[i-1] * (ADX_PERIOD - 1) + dx[i]) / ADX_PERIOD
    
    # Volume moving average
    volume_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < VOLUME_MA_PERIOD:
            volume_ma[i] = 0
        else:
            volume_ma[i] = np.mean(volume[i-VOLUME_MA_PERIOD:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla data not available
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or \
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
        
        # Determine market regime
        trending = adx[i] >= ADX_TREND_THRESHOLD
        ranging = adx[i] < ADX_TREND_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if volume_ma[i] > 0 else False
        
        # Price levels
        r4_level = r4_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        s4_level = s4_aligned[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if ranging:
            # Fade at S3/R3 in ranging markets
            long_entry = (low[i] <= s3_level and close[i] > s3_level) and volume_confirmed
            short_entry = (high[i] >= r3_level and close[i] < r3_level) and volume_confirmed
        else:
            # Breakout continuation in trending markets
            long_entry = (high[i] > r4_level and close[i] > r4_level) and volume_confirmed
            short_entry = (low[i] < s4_level and close[i] < s4_level) and volume_confirmed
        
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