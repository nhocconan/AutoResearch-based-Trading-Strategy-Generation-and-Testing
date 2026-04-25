#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: 12-hour Camarilla R1/S1 breakout with 1-day EMA50 trend filter and volume confirmation (>1.5x 20-period average).
Long when price breaks above R1 in 1-day uptrend with volume confirmation.
Short when price breaks below S1 in 1-day downtrend with volume confirmation.
Exit via ATR trailing stop (1.5*ATR from extreme) or opposite Camarilla level (S1 for long, R1 for short).
Camarilla levels provide precise intraday support/resistance derived from prior day's range.
Volume confirmation ensures breakouts have conviction. 1-day trend filter aligns with higher timeframe bias.
Designed for ~50-120 trades over 4 years (12-30/year) via tight breakout conditions.
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
    if len(df_1d) < 2:  # need at least 2 days for Camarilla
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d candle
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    # Using prior 1d candle's OHLC
    camarilla_range = high_1d - low_1d
    camarilla_r1 = close_1d + (camarilla_range * 1.1 / 12)
    camarilla_s1 = close_1d - (camarilla_range * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above R1 with volume confirmation
                long_signal = (close[i] > r1) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below S1 with volume confirmation
                short_signal = (close[i] < s1) and vol_regime[i]
            
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
            # 1. ATR trailing stop (1.5*ATR from extreme)
            atr_stop = long_extreme - 1.5 * atr[i]
            # 2. Price breaks below S1 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (1.5*ATR from extreme)
            atr_stop = short_extreme + 1.5 * atr[i]
            # 2. Price breaks above R1 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0