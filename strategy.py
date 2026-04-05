#!/usr/bin/env python3
"""
Experiment #7775: 6-hour Camarilla pivot reversion with weekly trend filter and volume confirmation.
Hypothesis: In ranging markets (common in bear cycles), price reverts from Camarilla R3/S3 levels; in trending markets, breaks through R4/S4 with volume continuation. Uses weekly trend to filter direction and avoid counter-trend trades. Works in both bull (buy R3/S3 in uptrend, break R4/S4) and bear (sell R3/S3 in downtrend, break R4/S4). Targets 60-120 trades over 4 years.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7775_6h_camarilla_pivot_weekly_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day for Camarilla calculation
WEEKLY_EMA = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=WEEKLY_EMA, adjust=False, min_periods=WEEKLY_EMA).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily OHLC for Camarilla pivots (use previous day's values)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6t
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]):
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
        
        # Determine weekly trend
        uptrend = ema_1w_aligned[i] > ema_1w_aligned[i-1]  # weekly EMA rising
        downtrend = ema_1w_aligned[i] < ema_1w_aligned[i-1]  # weekly EMA falling
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price levels
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Skip if pivot levels invalid
        if np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or np.isnan(s4):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Entry logic based on weekly trend
        if uptrend:
            # In uptrend: look for bounces off S3/S4 or breaks above R4
            long_bounce = (low[i] <= s3 and close[i] > s3) or (low[i] <= s4 and close[i] > s4)
            breakout = close[i] > r4
            long_entry = (long_bounce or breakout) and volume_confirmed
            
            if long_entry and position == 0:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
        
        elif downtrend:
            # In downtrend: look for bounces off R3/R4 or breaks below S4
            short_bounce = (high[i] >= r3 and close[i] < r3) or (high[i] >= r4 and close[i] < r4)
            breakdown = close[i] < s4
            short_entry = (short_bounce or breakdown) and volume_confirmed
            
            if short_entry and position == 0:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
        
        # Maintain position
        if position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
        else:
            signals[i] = 0.0
    
    return signals