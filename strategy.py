#!/usr/bin/env python3
"""
12h RSI Reversal with 1-day Volume Confirmation and Weekly Trend Filter
Look for RSI extremes combined with volume spikes and weekly trend alignment
Designed to work in both trending and ranging markets with low trade frequency
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 1-day average volume (24 periods of 12h = 1 day)
    vol_ma_1d = pd.Series(vol_1d).rolling(window=24, min_periods=24).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate RSI(14) on 12h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for RSI and averages
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike_1d_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + volume spike + above weekly EMA
            if (rsi_val < 30 and 
                vol_spike and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + volume spike + below weekly EMA
            elif (rsi_val > 70 and 
                  vol_spike and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or trend reversal
            if rsi_val >= 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or trend reversal
            if rsi_val <= 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSIReversal_VolumeSpike_WeeklyTrend"
timeframe = "12h"
leverage = 1.0