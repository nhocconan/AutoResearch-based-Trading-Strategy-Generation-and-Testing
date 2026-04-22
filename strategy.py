#!/usr/bin/env python3
"""
Hypothesis: 1-hour Volume-Weighted RSI with 4-hour Trend Filter and Volume Spike.
Long when VWRSI < 30, 4h EMA50 rising, and volume > 1.5x 20-period average.
Short when VWRSI > 70, 4h EMA50 falling, and volume > 1.5x 20-period average.
Exit when VWRSI crosses 50 or volume drops below average.
VWRSI reduces noise; volume spike confirms momentum; 4h EMA50 filters trend.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following 4h trend while using 1h VWRSI for entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4-hour data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume-weighted RSI (14-period)
    # Typical price
    tp = (high + low + close) / 3.0
    # Volume-weighted typical price
    vwtp = tp * volume
    # Sum of volume-weighted typical price and volume over window
    vwtp_sum = pd.Series(vwtp).rolling(window=14, min_periods=14).sum().values
    vol_sum = pd.Series(volume).rolling(window=14, min_periods=14).sum().values
    # Avoid division by zero
    vwtp_avg = np.divide(vwtp_sum, vol_sum, out=np.full_like(vwtp_sum, np.nan), where=vol_sum!=0)
    
    # Calculate price changes for RSI
    delta = vwtp - np.roll(vwtp, 1)
    delta[0] = 0
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    up_smoothed = pd.Series(up).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    down_smoothed = pd.Series(down).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # RSI calculation
    rs = np.divide(up_smoothed, down_smoothed, out=np.full_like(up_smoothed, np.nan), where=down_smoothed!=0)
    vwrsi = 100 - (100 / (1 + rs))
    
    # Volume spike detector: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(vwrsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vwrsi[i-1]) if i > 0 else False or
            np.isnan(ema50_4h_aligned[i-1]) if i > 0 else False):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VWRSI < 30, 4h EMA50 rising, and volume spike
            if (vwrsi[i] < 30 and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: VWRSI > 70, 4h EMA50 falling, and volume spike
            elif (vwrsi[i] > 70 and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: VWRSI crosses above 50 OR volume drops below average
                if (vwrsi[i] > 50 or 
                    volume[i] < vol_ma20[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: VWRSI crosses below 50 OR volume drops below average
                if (vwrsi[i] < 50 or 
                    volume[i] < vol_ma20[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_VWRSI_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0