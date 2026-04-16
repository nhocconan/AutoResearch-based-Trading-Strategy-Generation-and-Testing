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
    
    # === Daily OHLC for Donchian channel calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channel
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === ATR for volatility filter (14-period) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - low_1d[:-1])
    tr3 = np.abs(low_1d[1:] - high_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF data to 4h timeframe
    highest_20d_4h = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_4h = align_htf_to_ltf(prices, df_1d, lowest_20d)
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Volatility filter: ATR must be above its 50-period average ===
    atr_ma = pd.Series(atr_1d_4h).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_1d_4h > atr_ma
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20d_4h[i]) or np.isnan(lowest_20d_4h[i]) or
            np.isnan(atr_1d_4h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = highest_20d_4h[i]
        lower = lowest_20d_4h[i]
        vol_ok = vol_filter[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when price moves against position or volatility drops ===
        if position == 1:  # Long position
            # Exit when price drops below lower band or volatility drops
            if price < lower or not vol_ok:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above upper band or volatility drops
            if price > upper or not vol_ok:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band with volume spike and sufficient volatility
            if price > upper and vol_spike and vol_ok:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below lower band with volume spike and sufficient volatility
            elif price < lower and vol_spike and vol_ok:
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

name = "4h_Donchian20_VolumeSpike_VolFilter"
timeframe = "4h"
leverage = 1.0