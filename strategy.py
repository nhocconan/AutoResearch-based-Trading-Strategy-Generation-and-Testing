#!/usr/bin/env python3
"""
1h_VolumeSpike_CamarillaBreakout_4hTrendFilter
Hypothesis: 1-hour Camarilla R3/S3 breakout with 4-hour EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 in 4h uptrend (close > 4h EMA50) with volume > 2.0x 20-period average.
Short when price breaks below S3 in 4h downtrend (close < 4h EMA50) with volume > 2.0x 20-period average.
Exit via opposite Camarilla level (S3 for longs, R3 for shorts) or ATR stop (2.5*ATR from extreme).
Uses 4h for signal direction, 1h only for entry timing to target 15-37 trades/year.
Works in bull/bear markets via 4h EMA50 filter; avoids false breakouts via volume confirmation.
Session filter (08-20 UTC) reduces noise trades.
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter and Camarilla levels (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR for stoploss (14-period)
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
    
    # Calculate Camarilla levels from previous 4h bar (using 4h OHLC)
    # For each 4h bar, we need the previous 4h bar's OHLC
    # We'll calculate Camarilla levels for each 4h bar and align to 1h
    
    # Shift to get previous bar's OHLC
    o_4h_prev = np.roll(df_4h['open'].values, 1)
    h_4h_prev = np.roll(df_4h['high'].values, 1)
    l_4h_prev = np.roll(df_4h['low'].values, 1)
    c_4h_prev = np.roll(df_4h['close'].values, 1)
    # First value will be invalid (rolled from last), set to NaN
    o_4h_prev[0] = np.nan
    h_4h_prev[0] = np.nan
    l_4h_prev[0] = np.nan
    c_4h_prev[0] = np.nan
    
    # Calculate Camarilla levels for previous 4h bar
    # R3 = c + ((h-l)*1.1/4)
    # S3 = c - ((h-l)*1.1/4)
    camarilla_r3_4h = c_4h_prev + ((h_4h_prev - l_4h_prev) * 1.1 / 4)
    camarilla_s3_4h = c_4h_prev - ((h_4h_prev - l_4h_prev) * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_trend = ema_50_4h_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (4h EMA50 filter)
            if close[i] > ema_trend:  # 4h uptrend regime
                # Long: break above R3 with volume confirmation
                long_signal = (close[i] > r3_level) and vol_regime[i]
            else:  # 4h downtrend regime
                # Short: break below S3 with volume confirmation
                short_signal = (close[i] < s3_level) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.20
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.20
                position = -1
                short_extreme = close[i]
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
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Price breaks below S3 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s3_level:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above R3 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r3_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VolumeSpike_CamarillaBreakout_4hTrendFilter"
timeframe = "1h"
leverage = 1.0