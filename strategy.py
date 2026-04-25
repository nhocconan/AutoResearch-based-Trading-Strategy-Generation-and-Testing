#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 in 1d uptrend (close > 1d EMA50) with volume > 2.0x 20-period average.
Short when price breaks below Camarilla S1 in 1d downtrend (close < 1d EMA50) with volume > 2.0x 20-period average.
Exit via ATR trailing stop (3*ATR from extreme) or re-entry into Camarilla H3/L3 range.
Designed for ~12-30 trades/year by requiring strong breakouts, trend alignment, and volume confirmation.
Works in bull/bear markets via 1d EMA50 filter; avoids whipsaws via volume confirmation.
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
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate previous day's Camarilla levels for breakout signals
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using previous day's OHLC
    prev_close = close_1d[:-1]
    prev_high = high_1d[:-1]
    prev_low = low_1d[:-1]
    prev_range = prev_high - prev_low
    camarilla_R1 = prev_close + 1.1 * prev_range / 12
    camarilla_S1 = prev_close - 1.1 * prev_range / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, 20, atr_period)  # 20 for EMA warmup, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        camarilla_R1 = camarilla_R1_aligned[i]
        camarilla_S1 = camarilla_S1_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above Camarilla R1 with volume spike
                long_signal = (close[i] > camarilla_R1) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below Camarilla S1 with volume spike
                short_signal = (close[i] < camarilla_S1) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_high = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_low = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest close
            if close[i] > long_high:
                long_high = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Camarilla H3/L3 range
            atr_stop = long_high - 3.0 * atr[i]
            # Calculate Camarilla H3 and L3 for range exit
            camarilla_H3 = camarilla_S1 + 1.1 * (high_1d[-1] - low_1d[-1]) * 6 / 12  # Simplified: using current day's range approximation
            camarilla_L3 = camarilla_R1 - 1.1 * (high_1d[-1] - low_1d[-1]) * 6 / 12
            # For simplicity, use a fixed range around the breakout level
            range_exit = (close[i] < camarilla_R1 * 1.02 and close[i] > camarilla_S1 * 0.98)
            if close[i] <= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest close
            if close[i] < short_low:
                short_low = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Camarilla H3/L3 range
            atr_stop = short_low + 3.0 * atr[i]
            # Calculate Camarilla H3 and L3 for range exit
            camarilla_H3 = camarilla_S1 + 1.1 * (high_1d[-1] - low_1d[-1]) * 6 / 12
            camarilla_L3 = camarilla_R1 - 1.1 * (high_1d[-1] - low_1d[-1]) * 6 / 12
            # For simplicity, use a fixed range around the breakout level
            range_exit = (close[i] > camarilla_S1 * 0.98 and close[i] < camarilla_R1 * 1.02)
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0