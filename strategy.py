#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with 1w trend filter and volume confirmation.
# Fade at R3/S3 levels in ranging markets (1w ADX < 25), breakout continuation at R4/S4 in trending markets (1w ADX >= 25).
# Uses 1w ADX for regime detection, 1d Camarilla for entry/exit levels, and volume for confirmation.
# Designed for ~80-150 total trades over 4 years (20-38/year) to avoid excessive fees.
# Works in bull (breakouts at R4/S4 with volume) and bear (fades at R3/S3 with volume) markets.

name = "exp_13747_6h_camarilla1d_adx1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
VOLUME_MA_PERIOD = 6
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        if plus_dm[i] > minus_dm[i]:
            minus_dm[i] = 0
        elif minus_dm[i] > plus_dm[i]:
            plus_dm[i] = 0
        else:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    dx = np.zeros_like(high)
    
    for i in range(period, len(high)):
        if atr[i] != 0:
            plus_di[i] = (plus_dm[i] / atr[i]) * 100
            minus_di[i] = (minus_dm[i] / atr[i]) * 100
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    adx = np.zeros_like(high)
    adx_first = np.mean(dx[period:2*period]) if len(dx) >= 2*period else 0
    if len(adx) >= 2*period:
        adx[2*period-1] = adx_first
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    
    c = close
    h = high
    l = low
    
    r4 = c + ((h - l) * 1.5000)
    r3 = c + ((h - l) * 1.2500)
    r2 = c + ((h - l) * 1.1666)
    r1 = c + ((h - l) * 1.0833)
    
    s1 = c - ((h - l) * 1.0833)
    s2 = c - ((h - l) * 1.1666)
    s3 = c - ((h - l) * 1.2500)
    s4 = c - ((h - l) * 1.5000)
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data for filters ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w ADX for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, ADX_PERIOD)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's levels
    r1_1d = np.full_like(high_1d, np.nan)
    r2_1d = np.full_like(high_1d, np.nan)
    r3_1d = np.full_like(high_1d, np.nan)
    r4_1d = np.full_like(high_1d, np.nan)
    s1_1d = np.full_like(high_1d, np.nan)
    s2_1d = np.full_like(high_1d, np.nan)
    s3_1d = np.full_like(high_1d, np.nan)
    s4_1d = np.full_like(high_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        r1_1d[i] = r1
        r2_1d[i] = r2
        r3_1d[i] = r3
        r4_1d[i] = r4
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate ATR for stop loss (using 6h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA for 6h
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD*2, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_1w_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or \
           np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or \
           np.isnan(s2_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(volume_ma[i]):
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
        
        # Volume confirmation (using 6h volume)
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Regime detection from 1w ADX
        is_trending = adx_1w_aligned[i] >= ADX_TREND_THRESHOLD
        is_ranging = adx_1w_aligned[i] < ADX_TREND_THRESHOLD
        
        # Initialize signal flags
        long_signal = False
        short_signal = False
        
        if is_ranging:
            # Ranging market: fade at R3/S3
            # Long when price touches S3 and shows rejection
            if i > 0:
                touched_s3 = low[i] <= s3_1d_aligned[i] and close[i] > s3_1d_aligned[i]
                rejected_s3 = touched_s3 and close[i] > open[i]  # bullish candle
                long_signal = volume_ok and rejected_s3
                
                # Short when price touches R3 and shows rejection
                touched_r3 = high[i] >= r3_1d_aligned[i] and close[i] < r3_1d_aligned[i]
                rejected_r3 = touched_r3 and close[i] < open[i]  # bearish candle
                short_signal = volume_ok and rejected_r3
        else:
            # Trending market: breakout continuation at R4/S4
            # Long when price breaks above R4 with volume
            if i > 0:
                broke_r4 = close[i] > r4_1d_aligned[i] and close[i-1] <= r4_1d_aligned[i-1]
                long_signal = volume_ok and broke_r4
                
                # Short when price breaks below S4 with volume
                broke_s4 = close[i] < s4_1d_aligned[i] and close[i-1] >= s4_1d_aligned[i-1]
                short_signal = volume_ok and broke_s4
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            if is_ranging:
                # Exit long when price approaches R3 in ranging market
                if high[i] >= r3_1d_aligned[i]:
                    exit_signal = True
            else:
                # Exit long when price approaches S4 in trending market (trailing stop)
                if close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            if is_ranging:
                # Exit short when price approaches S3 in ranging market
                if low[i] <= s3_1d_aligned[i]:
                    exit_signal = True
            else:
                # Exit short when price approaches R4 in trending market (trailing stop)
                if close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals