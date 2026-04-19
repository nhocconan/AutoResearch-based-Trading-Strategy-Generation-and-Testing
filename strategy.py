#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d EMA200 trend filter.
# Long when price breaks above Donchian high (20) + volume > 1.5x 20-period average + price > 1d EMA200
# Short when price breaks below Donchian low (20) + volume > 1.5x 20-period average + price < 1d EMA200
# Exit when price crosses back to 1d EMA200 or Donchian midpoint.
# Target: 20-40 trades/year per symbol with strong trend capture and low whipsaw.
name = "4h_Donchian20_EMA200_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Donchian channels (20-period)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    donch_mid = np.zeros(n)
    
    for i in range(n):
        if i < 19:
            donch_high[i] = np.nan
            donch_low[i] = np.nan
        else:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
            donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # Align 1d EMA200 to 4h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 19  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200_val = ema_200_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above Donchian high + volume + above EMA200
            if price > donch_high_val and volume_confirmed and price > ema_200_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + volume + below EMA200
            elif price < donch_low_val and volume_confirmed and price < ema_200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA200 or Donchian midpoint
            if price < ema_200_val or price < donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA200 or Donchian midpoint
            if price > ema_200_val or price > donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals