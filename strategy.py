#!/usr/bin/env python3
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
    
    # === 12h price (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12-period ATR for volatility filter (12h timeframe)
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=12, min_periods=12).mean().values
    
    # Align 12h data to primary timeframe (12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # === Daily ATR and price channel ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR for volatility regime filter
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels on daily
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily data to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === Daily volume spike detection ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_12h_aligned[i]
        upper_level = upper_20_aligned[i]
        lower_level = lower_20_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 12h EMA(20) or hits 1.5x ATR stop
            ema_20_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
            ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
            if price < ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 12h EMA(20) or hits 1.5x ATR stop
            ema_20_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
            ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
            if price > ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above daily upper Donchian with 12h volume spike and low volatility regime
            if (price > upper_level and 
                volume[i] > 1.5 * np.median(volume[max(0, i-50):i+1]) and  # 12h volume spike
                atr_12h_val < atr_1d_val * 1.2):  # Low volatility regime (12h ATR < 1.2x daily ATR)
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below daily lower Donchian with 12h volume spike and low volatility regime
            elif (price < lower_level and 
                  volume[i] > 1.5 * np.median(volume[max(0, i-50):i+1]) and  # 12h volume spike
                  atr_12h_val < atr_1d_val * 1.2):  # Low volatility regime (12h ATR < 1.2x daily ATR)
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_DonchianBreakout_VolumeSpike_LowVol"
timeframe = "12h"
leverage = 1.0