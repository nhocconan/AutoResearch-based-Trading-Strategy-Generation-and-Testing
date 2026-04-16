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
    
    # === Daily data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-day ATR for volatility
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels on daily
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === Weekly data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-week EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === Daily volume spike detection ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Align all HTF data to daily timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]  # Use daily close for entry/exit logic
        upper_level = upper_20_aligned[i]
        lower_level = lower_20_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly EMA(50)
            if price < ema_50_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly EMA(50)
            if price > ema_50_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above daily upper Donchian with volume spike and bullish weekly trend
            if (price > upper_level and 
                volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and  # Daily volume spike
                price > ema_50_val):  # Bullish weekly trend filter
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below daily lower Donchian with volume spike and bearish weekly trend
            elif (price < lower_level and 
                  volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and  # Daily volume spike
                  price < ema_50_val):  # Bearish weekly trend filter
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

name = "1d_DonchianBreakout_VolumeSpike_WeeklyTrend"
timeframe = "1d"
leverage = 1.0