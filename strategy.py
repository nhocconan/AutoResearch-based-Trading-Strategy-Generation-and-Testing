#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability reversal/continuation points
# Breakout above R3 or below S3 with volume confirmation indicates strong institutional participation
# 1d EMA34 filter ensures alignment with daily trend for higher win rate
# Volume spike (>2.0x 20-period EMA) filters weak breakouts
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag while capturing strong moves

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #          S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use the previous day's OHLC to calculate today's levels
    if len(df_1d) >= 2:
        prev_high = df_1d['high'].iloc[-2]  # Previous day high
        prev_low = df_1d['low'].iloc[-2]    # Previous day low
        prev_close = df_1d['close'].iloc[-2] # Previous day close
        
        # Calculate Camarilla levels for current day
        diff = prev_high - prev_low
        camarilla_r3 = prev_close + 1.125 * diff
        camarilla_s3 = prev_close - 1.125 * diff
    else:
        camarilla_r3 = camarilla_s3 = 0
    
    # Broadcast Camarilla levels to all 4h bars (same for entire day)
    camarilla_r3_4h = np.full(n, camarilla_r3)
    camarilla_s3_4h = np.full(n, camarilla_s3)
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 + above 1d EMA34 + volume spike
            if (close[i] > camarilla_r3_4h[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below Camarilla S3 + below 1d EMA34 + volume spike
            elif (close[i] < camarilla_s3_4h[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price closes below Camarilla R3 OR below 1d EMA34
            if close[i] < camarilla_r3_4h[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price closes above Camarilla S3 OR above 1d EMA34
            if close[i] > camarilla_s3_4h[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals