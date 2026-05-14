#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_1dTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need at least 10 days of 4h data
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 days for Donchian and EMA
        return np.zeros(n)
    
    # === 1d: Calculate 20-day trend and Donchian channels (using previous day's data) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # 20-day EMA for trend filter
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 20-day Donchian channels (highest high, lowest low over 20 days)
    # We need to calculate these manually since we're using previous day's data
    donchian_high = np.full_like(high_1d, np.nan)
    donchian_low = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Align 1d indicators to 4h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
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
            # Long: Price above 20-day EMA (uptrend), breaks above 20-day Donchian high, volume confirmation
            if (close_val > ema_val and  # Uptrend filter
                close_val > upper_donchian and   # Break above Donchian high
                vol_ratio_val > 1.8):    # Increased volume confirmation threshold
                signals[i] = 0.25
                position = 1
            # Short: Price below 20-day EMA (downtrend), breaks below 20-day Donchian low, volume confirmation
            elif (close_val < ema_val and  # Downtrend filter
                  close_val < lower_donchian and   # Break below Donchian low
                  vol_ratio_val > 1.8):    # Increased volume confirmation threshold
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below 20-day EMA or breaks below 20-day Donchian low (reversal)
            if close_val < ema_val or close_val < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above 20-day EMA or breaks above 20-day Donchian high (reversal)
            if close_val > ema_val or close_val > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals