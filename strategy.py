#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Enters long when price breaks above 1d Donchian high AND 1w close > 1w SMA50 (uptrend)
# Enters short when price breaks below 1d Donchian low AND 1w close < 1w SMA50 (downtrend)
# Volume confirmation: current volume > 1.5x 20-period average on 1d
# Exits on opposite Donchian break or trend reversal
# Position size 0.25 to manage drawdown
# Target: 10-25 trades/year per symbol (40-100 total over 4 years)

name = "1d_1w_donchian_trend_vol_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donch_high_1d = np.full(len(df_1d), np.nan)
    donch_low_1d = np.full(len(df_1d), np.nan)
    
    for i in range(20, len(df_1d)):
        donch_high_1d[i] = np.max(high_1d[i-20:i])
        donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Align 1d Donchian to 1d timeframe (only use completed daily bars)
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 1w SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = np.full(len(df_1w), np.nan)
    for i in range(50, len(df_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-50:i])
    
    # Align 1w SMA50 to 1d timeframe (only use completed weekly bars)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume confirmation: 20-period average on 1d
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after SMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_1d_aligned[i]) or 
            np.isnan(donch_low_1d_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        if position == 1:  # Long position
            # Exit: price closes below 1d Donchian low OR trend turns down
            if close[i] <= donch_low_1d_aligned[i] or close_1w[-1] < sma_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d Donchian high OR trend turns up
            if close[i] >= donch_high_1d_aligned[i] or close_1w[-1] > sma_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 1d Donchian high with volume + uptrend
            if (close[i] > donch_high_1d_aligned[i] and 
                vol_ratio > 1.5 and 
                close_1w[-1] > sma_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 1d Donchian low with volume + downtrend
            elif (close[i] < donch_low_1d_aligned[i] and 
                  vol_ratio > 1.5 and 
                  close_1w[-1] < sma_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals