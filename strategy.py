#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation.
# Long when price breaks above R1 with uptrend (price > 4h EMA50) and volume > 1.5x 20-bar average.
# Short when price breaks below S1 with downtrend (price < 4h EMA50) and volume spike.
# Uses ATR trailing stop (2.0x) for risk management and session filter (08-20 UTC).
# Targets 60-150 total trades over 4 years (15-37/year) with discrete position sizing (0.20).
# Works in both bull/bear markets by requiring 4h EMA50 trend alignment and volume confirmation.
# Uses 4h HTF for Camarilla levels and trend filter to reduce noise and look-ahead bias.
# 1h timeframe chosen for better entry timing while using 4h for signal direction.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_ATRTrail_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from 4h OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    rng = high_4h - low_4h
    camarilla_r1 = close_4h_arr + (rng * 1.1 / 12)
    camarilla_s1 = close_4h_arr - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            continue
        
        # Skip if Camarilla levels not available
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            continue
        
        # Regime filter: price above/below 4h EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend and curr_close > r1_aligned[i] and curr_volume_spike:
                signals[i] = 0.20
                position = 1
                highest_since_entry = curr_close
            elif is_downtrend and curr_close < s1_aligned[i] and curr_volume_spike:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals