#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR-based breakout with volume confirmation and weekly trend filter.
# Uses ATR-based breakout levels (similar to Donchian but volatility-adjusted) to capture
# breakouts in both trending and ranging markets. Volume confirmation ensures breakouts
# have conviction. Weekly trend filter avoids counter-trend trades. Designed for 4-8
# trades per month (~48-96/year) to minimize fee drag while capturing significant moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # ATR-based breakout levels (20-period)
    atr_period = 20
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # ATR-based channels: midpoint ± ATR multiplier
    avg_price = (high + low) / 2
    atr_mult = 1.5
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    for i in range(atr_period, n):
        avg = np.mean(avg_price[i-atr_period+1:i+1])
        upper_band[i] = avg + atr_mult * atr[i]
        lower_band[i] = avg - atr_mult * atr[i]
    
    # Volume confirmation (20-period average)
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Weekly trend filter: 50-period SMA
    close_1w = df_1w['close'].values
    sma_1w = np.zeros(len(close_1w))
    for i in range(50, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i-50:i])
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(sma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = upper_band[i]
        lower = lower_band[i]
        atr_val = atr[i]
        weekly_sma = sma_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # ATR filter: ATR > 0 (always true, but keeps structure)
        atr_filter = atr_val > 0
        
        if position == 0:
            # Long: price breaks above upper band + volume + price above weekly SMA
            if (price > upper and 
                volume_confirm and 
                atr_filter and
                price > weekly_sma):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower band + volume + price below weekly SMA
            elif (price < lower and 
                  volume_confirm and 
                  atr_filter and
                  price < weekly_sma):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower band OR volatility drops significantly
            if (price < lower or 
                atr_val < 0.5 * np.mean(atr[max(0, i-20):i+1])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper band OR volatility drops significantly
            if (price > upper or 
                atr_val < 0.5 * np.mean(atr[max(0, i-20):i+1])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1w_ATR_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0