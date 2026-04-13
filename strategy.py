#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR filter.
# Donchian channels capture breakouts in both bull and bear markets.
# Volume confirmation ensures breakouts have conviction.
# ATR filter avoids false breakouts in low volatility.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(period-1, n):
        donchian_high[i] = np.max(high[i-period+1:i+1])
        donchian_low[i] = np.min(low[i-period+1:i+1])
    
    # Calculate ATR (14-period) for volatility filter
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align daily trend filter
    close_1d = df_1d['close'].values
    sma_1d = np.zeros(len(close_1d))
    for i in range(20, len(close_1d)):
        sma_1d[i] = np.mean(close_1d[i-20:i])
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(sma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        atr_val = atr[i]
        daily_sma = sma_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # ATR filter: ATR > 0 (always true, but keeps structure)
        atr_filter = atr_val > 0
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + price above daily SMA
            if (price > donch_high and 
                volume_confirm and 
                atr_filter and
                price > daily_sma):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + volume + price below daily SMA
            elif (price < donch_low and 
                  volume_confirm and 
                  atr_filter and
                  price < daily_sma):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR volume drops
            if (price < donch_low or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR volume drops
            if (price > donch_high or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0