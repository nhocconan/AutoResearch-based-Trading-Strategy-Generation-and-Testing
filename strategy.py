#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 in 1d uptrend (close > 1d EMA50) with volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 in 1d downtrend (close < 1d EMA50) with volume > 2.0x 20-period average.
Exit via ATR-based trailing stop (3*ATR from extreme) or re-entry into Camarilla H3/L3 range.
Designed for ~12-37 trades/year by requiring strong breakouts and volume confirmation.
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
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: based on previous day's high, low, close
    # R4 = Close + ((High-Low) * 1.1/2)
    # R3 = Close + ((High-Low) * 1.1/4)
    # R2 = Close + ((High-Low) * 1.1/6)
    # R1 = Close + ((High-Low) * 1.1/12)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High-Low) * 1.1/12)
    # S2 = Close - ((High-Low) * 1.1/6)
    # S3 = Close - ((High-Low) * 1.1/4)
    # S4 = Close - ((High-Low) * 1.1/2)
    
    # Need previous day's data - we'll use HTF data for this
    # For 12h timeframe, we need to access previous 1d bar's HLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_H3 = close_1d + ((high_1d - low_1d) * 1.1 / 6)  # H3 for exit
    camarilla_L3 = close_1d - ((high_1d - low_1d) * 1.1 / 6)  # L3 for exit
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # ATR for trailing stop (14-period)
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
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above Camarilla R3 with volume spike
                long_signal = (close[i] > camarilla_R3_aligned[i]) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below Camarilla S3 with volume spike
                short_signal = (close[i] < camarilla_S3_aligned[i]) and vol_regime[i]
            
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
            range_exit = (close[i] < camarilla_H3_aligned[i] and close[i] > camarilla_L3_aligned[i])
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
            range_exit = (close[i] > camarilla_L3_aligned[i] and close[i] < camarilla_H3_aligned[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0