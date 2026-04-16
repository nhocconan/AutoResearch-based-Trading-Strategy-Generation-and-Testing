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
    
    # === Daily OHLC for ATR calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === ATR (14-period) for volatility filter ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # === 12h Donchian channel (20-period) for breakout ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF data to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_12h, donchian_low)
    atr_1d_avg_12h = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or
            np.isnan(atr_1d_avg_12h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high_12h[i]
        lower = donchian_low_12h[i]
        atr_avg = atr_1d_avg_12h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when price moves against position or volatility drops ===
        if position == 1:  # Long position
            # Exit when price drops below lower band or volatility drops significantly
            if price < lower or atr_avg < (atr_1d_avg_12h[i-1] * 0.7 if i > 0 else atr_avg):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above upper band or volatility drops significantly
            if price > upper or atr_avg < (atr_1d_avg_12h[i-1] * 0.7 if i > 0 else atr_avg):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band with volume spike and sufficient volatility
            if price > upper and vol_spike and atr_avg > 0:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below lower band with volume spike and sufficient volatility
            elif price < lower and vol_spike and atr_avg > 0:
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