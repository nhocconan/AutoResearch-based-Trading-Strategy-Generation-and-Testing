#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_DonchianBreakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h: Trend filter (20-period EMA) ===
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 12h: Donchian channels (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high_12h = np.full_like(high_12h, np.nan)
    donchian_low_12h = np.full_like(low_12h, np.nan)
    
    for i in range(20, len(high_12h)):
        donchian_high_12h[i] = np.max(high_12h[i-20:i])
        donchian_low_12h[i] = np.min(low_12h[i-20:i])
    
    # === Align 12h indicators to 6h timeframe ===
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        ema_val = ema_20_12h_aligned[i]
        upper_donchian = donchian_high_12h_aligned[i]
        lower_donchian = donchian_low_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(upper_donchian) or 
            np.isnan(lower_donchian) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above 12h EMA (uptrend), breaks above 12h Donchian high, volume confirmation
            if (close_val > ema_val and  # Uptrend filter
                close_val > upper_donchian and   # Break above Donchian high
                vol_ratio_val > 1.5):    # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price below 12h EMA (downtrend), breaks below 12h Donchian low, volume confirmation
            elif (close_val < ema_val and  # Downtrend filter
                  close_val < lower_donchian and   # Break below Donchian low
                  vol_ratio_val > 1.5):    # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below 12h EMA or breaks below 12h Donchian low (reversal)
            if close_val < ema_val or close_val < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above 12h EMA or breaks above 12h Donchian high (reversal)
            if close_val > ema_val or close_val > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals