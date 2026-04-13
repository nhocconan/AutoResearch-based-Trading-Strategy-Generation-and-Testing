#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h Donchian breakout (20-period) + volume confirmation + 12h EMA trend filter.
# Long: Price breaks above Donchian(20) high + volume > 1.5x avg volume + price > 12h EMA(50).
# Short: Price breaks below Donchian(20) low + volume > 1.5x avg volume + price < 12h EMA(50).
# Uses 12h Donchian for structure, 4h for execution with volume and trend confirmation.
# Exit: Opposite Donchian break or trend reversal. Target: 75-200 trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Donchian and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian channels (20-period) on 12h
    donch_high = np.full(len(high_12h), np.nan)
    donch_low = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donch_high[i] = np.max(high_12h[i-20:i])
        donch_low[i] = np.min(low_12h[i-20:i])
    
    # EMA(50) on 12h close
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (20-period) on 4h
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 12h indicators to 4h
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        ema = ema_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above Donchian high + volume + above EMA
            if (price > upper and volume_confirm and price > ema):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian low + volume + below EMA
            elif (price < lower and volume_confirm and price < ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: break below Donchian low OR price below EMA
            if price < lower or price < ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: break above Donchian high OR price above EMA
            if price > upper or price > ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_EMA_Volume_v2"
timeframe = "4h"
leverage = 1.0