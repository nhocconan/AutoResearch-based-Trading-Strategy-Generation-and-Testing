#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily OHLC for calculations ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 12h OHLC for price action ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === Calculate 12h ATR (14-period) for volatility ===
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_slow = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    
    # === Calculate daily volatility filter ===
    tr1d = high_1d[1:] - low_1d[1:]
    tr2d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3d = np.abs(low_1d[1:] - close_1d[:-1])
    trd = np.maximum(tr1d, np.maximum(tr2d, tr3d))
    trd = np.concatenate([[np.nan], trd])
    
    atr_1d = pd.Series(trd).rolling(window=14, min_periods=14).mean().values
    atr_1d_slow = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # === Daily volume average for spike detection ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === Align all 1d data to 12h timeframe ===
    atr_1d_slow_12h = align_htf_to_ltf(prices, df_1d, atr_1d_slow)
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === Calculate 12h price channels (Donchian 20) ===
    def donchian_channels(high_arr, low_arr, window):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    dc_upper_12h, dc_lower_12h = donchian_channels(high_12h, low_12h, 20)
    
    # === Align 12h Donchian channels to lower timeframe ===
    dc_upper_aligned = align_htf_to_ltf(prices, df_12h, dc_upper_12h)
    dc_lower_aligned = align_htf_to_ltf(prices, df_12h, dc_lower_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper_aligned[i]) or np.isnan(dc_lower_aligned[i]) or
            np.isnan(atr_1d_slow_12h[i]) or np.isnan(vol_ma_1d_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = dc_upper_aligned[i]
        lower = dc_lower_aligned[i]
        atr_filter = atr_1d_slow_12h[i]
        vol_ma_filter = vol_ma_1d_12h[i]
        
        # Volume spike detection: current 12h volume > 2x 1d average volume
        # Get the corresponding 12h volume for volume spike check
        vol_12h_idx = i // 12  # Convert 12h index to approximate daily index for volume
        if vol_12h_idx >= len(volume):
            vol_spike = False
        else:
            vol_12h = volume[vol_12h_idx * 12:(vol_12h_idx + 1) * 12].sum() if (vol_12h_idx + 1) * 12 <= len(volume) else 0
            vol_spike = vol_12h > (2.0 * vol_ma_filter)
        
        # === EXIT LOGIC: Exit when price moves against position or volatility drops ===
        if position == 1:  # Long position
            # Exit when price drops below lower Donchian or volatility drops significantly
            if price < lower or atr_filter < (atr_1d_slow_12h[i-1] * 0.7 if i > 0 else atr_filter):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above upper Donchian or volatility drops significantly
            if price > upper or atr_filter < (atr_1d_slow_12h[i-1] * 0.7 if i > 0 else atr_filter):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian with volume spike and sufficient volatility
            if price > upper and vol_spike and atr_filter > 0:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below lower Donchian with volume spike and sufficient volatility
            elif price < lower and vol_spike and atr_filter > 0:
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

name = "12h_Donchian20_VolumeSpike_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0