#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Donchian20_1wTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1w: Calculate 20-week trend and Donchian channels (using previous week's data) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use previous week's OHLC for today's levels
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    
    # Set first week's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # 20-week EMA for trend filter
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 20-week Donchian channels (highest high, lowest low over 20 weeks)
    donchian_high = np.full_like(high_1w, np.nan)
    donchian_low = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Align 1w indicators to 12h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        ema_val = ema_20_aligned[i]
        upper_donchian = donchian_high_aligned[i]
        lower_donchian = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(upper_donchian) or 
            np.isnan(lower_donchian) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above 20-week EMA (uptrend), breaks above 20-week Donchian high, volume confirmation
            if (close_val > ema_val and  # Uptrend filter
                close_val > upper_donchian and   # Break above Donchian high
                vol_ratio_val > 1.8):    # Increased volume confirmation threshold
                signals[i] = 0.25
                position = 1
            # Short: Price below 20-week EMA (downtrend), breaks below 20-week Donchian low, volume confirmation
            elif (close_val < ema_val and  # Downtrend filter
                  close_val < lower_donchian and   # Break below Donchian low
                  vol_ratio_val > 1.8):    # Increased volume confirmation threshold
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below 20-week EMA or breaks below 20-week Donchian low (reversal)
            if close_val < ema_val or close_val < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above 20-week EMA or breaks above 20-week Donchian high (reversal)
            if close_val > ema_val or close_val > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals