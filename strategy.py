#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA200 trend filter and 1d volume confirmation
# Targets 1h timeframe with strict entry conditions using 4h trend alignment and 1d volume spike.
# Uses 0.20 position sizing to manage drawdown. Session filter (08-20 UTC) reduces noise.
# Designed for BTC/ETH with limited trades to overcome fee drag in ranging/bear markets.

name = "1h_Camarilla_R3S3_Breakout_4hEMA200_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA200 for trend filter (loaded ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 4h Camarilla pivot levels (R3, S3)
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_range = high_4h - low_4h
    r3 = close_4h + camarilla_range * 1.1 / 4
    s3 = close_4h - camarilla_range * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Calculate 1d volume confirmation (loaded ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for 4h EMA200)
    start_idx = 200  # 4h EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 2.0x 1d average volume
        volume_spike = volume[i] > (2.0 * vol_ma_1d_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike AND price > 4h EMA200 (bullish trend)
            if (close[i] > r3_aligned[i] and 
                volume_spike and 
                close[i] > ema_200_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume spike AND price < 4h EMA200 (bearish trend)
            elif (close[i] < s3_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_200_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 (breakdown) OR price below 4h EMA200 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 (breakout) OR price above 4h EMA200 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals