#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_VolumeSpike
Hypothesis: 1-hour Camarilla R3/S3 breakout with 4-hour trend filter (price > 4h EMA50) and volume confirmation (>2.0x 20-period average).
Long when price breaks above R3 in 4h uptrend with volume confirmation.
Short when price breaks below S3 in 4h downtrend with volume confirmation.
Exit via opposite Camarilla level (S3 for long, R3 for short) or ATR trailing stop (1.5*ATR from extreme).
Uses 4h for signal direction, 1h only for entry timing to minimize trades.
Target: 15-35 trades/year (60-140 over 4 years) via tight Camarilla breakout conditions.
Session filter: 08-20 UTC to reduce noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter and Camarilla calculation (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # need sufficient data for EMA50 and Camarilla
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels on 4h data (based on previous day's OHLC)
    # Camarilla uses previous day's range to calculate support/resistance levels
    # We'll use 4h data but calculate based on daily OHLC aggregated from 4h
    # For simplicity, we use 4h OHLC directly with a lookback of 6 periods (24h/4h=6)
    lookback = 6  # 6*4h = 24h approx
    if len(df_4h) < lookback:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar using previous 6 bars' OHLC
    # R3 = Close + 1.1*(High-Low)
    # S3 = Close - 1.1*(High-Low)
    # where High, Low, Close are from the prior lookback period
    high_max = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().values
    low_min = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().values
    close_prev = pd.Series(close_4h).shift(1).values  # previous bar close
    
    # Camarilla R3 and S3
    r3 = close_prev + 1.1 * (high_max - low_min)
    s3 = close_prev - 1.1 * (high_max - low_min)
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, lookback)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_trend = ema_50_4h_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (4h EMA50 filter)
            if close[i] > ema_trend:  # 4h uptrend regime
                # Long: break above R3 with volume confirmation
                long_signal = (close[i] > r3) and vol_regime[i]
            else:  # 4h downtrend regime
                # Short: break below S3 with volume confirmation
                short_signal = (close[i] < s3) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.20
                position = 1
                long_extreme = close[i]
                if 'long_signal' in locals(): del long_signal
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.20
                position = -1
                short_extreme = close[i]
                if 'short_signal' in locals(): del short_signal
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (1.5*ATR from extreme)
            atr_stop = long_extreme - 1.5 * atr[i]
            # 2. Price breaks below S3 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s3:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (1.5*ATR from extreme)
            atr_stop = short_extreme + 1.5 * atr[i]
            # 2. Price breaks above R3 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r3:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0