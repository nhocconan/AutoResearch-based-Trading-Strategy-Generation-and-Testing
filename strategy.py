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
    
    # === 1d RSI(14) for regime detection ===
    df_1d = get_htf_data(prices, '1d')
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # === 1h Donchian Channel (20-period) ===
    df_1h = get_htf_data(prices, '1h')
    donchian_high = pd.Series(df_1h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1h, donchian_low)
    
    # === Volume Confirmation (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)  # Strong volume spike
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100  # Need RSI and data alignment
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        rsi = rsi_1d_aligned[i]
        
        # === EXIT LOGIC: Exit when price returns to midline (average of Donchian) ===
        midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        
        if position == 1:  # Long position
            # Exit when price crosses back below midline (failed bullish continuation)
            if price < midline:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above midline (failed bearish continuation)
            if price > midline:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high with volume confirmation and RSI < 50 (bullish regime)
            if price > donchian_high_aligned[i] and vol_spike and rsi < 50:
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian low with volume confirmation and RSI > 50 (bearish regime)
            elif price < donchian_low_aligned[i] and vol_spike and rsi > 50:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_1d_RSI50_Volume"
timeframe = "1h"
leverage = 1.0