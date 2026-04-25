#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12-hour Camarilla R3/S3 level breakout with 1-day trend filter (price > 1d EMA50) and volume confirmation (>1.8x 20-period average).
Long when price breaks above R3 in 1-day uptrend with volume confirmation.
Short when price breaks below S3 in 1-day downtrend with volume confirmation.
Exit via opposite Camarilla level (S3 for long, R3 for short) or ATR trailing stop (2.0*ATR from extreme).
Camarilla levels provide precise intraday support/resistance derived from prior day's range.
Volume confirmation ensures breakouts have conviction. 1-day trend filter aligns with higher timeframe bias.
Designed for ~50-150 trades over 4 years (12-37/year) via tight Camarilla breakout conditions.
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
    
    # Get 1d data for trend filter and Camarilla calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # need at least 2 days for Camarilla calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla levels: based on prior day's (high-low) range
    # R4 = close + (high-low) * 1.1/2
    # R3 = close + (high-low) * 1.1/4
    # S3 = close - (high-low) * 1.1/4
    # S4 = close - (high-low) * 1.1/2
    # We use R3 and S3 as primary breakout levels
    prior_high = np.roll(high_1d, 1)  # prior day's high
    prior_low = np.roll(low_1d, 1)    # prior day's low
    prior_close = np.roll(close_1d, 1) # prior day's close
    
    # Handle first bar (no prior day)
    prior_high[0] = high_1d[0]
    prior_low[0] = low_1d[0]
    prior_close[0] = close_1d[0]
    
    range_1d = prior_high - prior_low
    camarilla_r3 = prior_close + (range_1d * 1.1 / 4)
    camarilla_s3 = prior_close - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above R3 with volume confirmation
                long_signal = (close[i] > r3) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below S3 with volume confirmation
                short_signal = (close[i] < s3) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.0*ATR from extreme)
            atr_stop = long_extreme - 2.0 * atr[i]
            # 2. Price breaks below S3 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s3:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.0*ATR from extreme)
            atr_stop = short_extreme + 2.0 * atr[i]
            # 2. Price breaks above R3 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r3:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0