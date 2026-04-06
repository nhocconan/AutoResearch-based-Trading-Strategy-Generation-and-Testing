#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Uses 12h price chart for entry/exit timing, with 1d Donchian channel (20-period) for trend direction.
# 1d EMA(50) as additional trend filter to avoid counter-trend trades.
# Volume confirmation: current 12h volume > 1.5x 20-period average to filter weak breakouts.
# Designed to capture medium-term trends in both bull and bear markets by aligning with higher timeframe direction.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_donchian20_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period) for entry timing
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # 1d Donchian channel (20-period) for trend direction (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_high_1d = np.full(len(close_1d), np.nan)
    donchian_low_1d = np.full(len(close_1d), np.nan)
    for i in range(19, len(close_1d)):
        donchian_high_1d[i] = np.max(high_1d[i-19:i+1])
        donchian_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # 1d EMA(50) for additional trend filter
    ema_50_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 49:
            ema_50_1d[i] = np.nan
        elif i == 49:
            ema_50_1d[i] = np.mean(close_1d[0:50])
        else:
            ema_50_1d[i] = close_1d[i] * 2/(50+1) + ema_50_1d[i-1] * (1 - 2/(50+1))
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if HTF data not available
        if np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below 12h Donchian low or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < donchian_low[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 12h Donchian high or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > donchian_high[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 1d trend filter
            if volume_filter:
                # Breakout above 12h Donchian high with 1d uptrend
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and 
                    close[i] > donchian_high_1d_aligned[i] and close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below 12h Donchian low with 1d downtrend
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and 
                      close[i] < donchian_low_1d_aligned[i] and close[i] < ema_50_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals