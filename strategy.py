#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and daily EMA50 trend filter
# Long when price breaks above Donchian(20) high + volume > 1.5x avg + price > daily EMA50
# Short when price breaks below Donchian(20) low + volume > 1.5x avg + price < daily EMA50
# Exit when price crosses Donchian midline or volume dries up
# Designed to work in both bull and bear markets via trend filter
# Target: ~25-40 trades/year per symbol (~100-160 total over 4 years)

name = "4h_DonchianBreakout_Volume_EMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_mid = (high_max + low_min) / 2
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian channel data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max.iloc[i]) or np.isnan(low_min.iloc[i]) or 
            np.isnan(donchian_mid.iloc[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_max.iloc[i]
        lower = low_min.iloc[i]
        midline = donchian_mid.iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper band + volume + above daily EMA50
            if price > upper and volume_confirmed and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band + volume + below daily EMA50
            elif price < lower and volume_confirmed and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below midline OR volume dries up
            if price < midline or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above midline OR volume dries up
            if price > midline or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals