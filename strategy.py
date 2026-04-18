#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d trend filter.
# Uses 4h Donchian (20) for breakout direction, 1d EMA50 for trend filter, and volume spike for confirmation.
# Designed for low trade frequency (target 15-37/year) to avoid fee drag in both bull and bear markets.
# Works in bull markets via breakout continuation and in bear markets via mean reversion off bands.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channel (20-period)
    donch_high = np.full(len(high_4h), np.nan)
    donch_low = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        donch_high[i] = np.max(high_4h[i-20:i])
        donch_low[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50_1d[i-1] * (49 / (50 + 1)))
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 20)  # need 4h Donchian, 1d EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian high with volume and trend filter
            if (close[i] > donch_high_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian low with volume and trend filter
            elif (close[i] < donch_low_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 4h Donchian low or opposite Donchian breakout
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h Donchian high or opposite Donchian breakout
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hBreakout_Volume_1dEMA50"
timeframe = "1h"
leverage = 1.0