#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price channel breakout (Donchian 20) with 1d EMA50 filter and volume confirmation.
# Uses Donchian channels to capture breakouts in trending markets.
# 1d EMA50 acts as a trend filter to avoid counter-trend trades.
# Volume spike (>2x 20-period average) confirms breakout conviction.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels on 12h data (20-period high/low)
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    
    donchian_high = high_12h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_12h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe (12h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d data
    close_1d = pd.Series(df_1d['close'].values)
    ema_50 = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to lower timeframe (12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        ema50 = ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian band AND above EMA50 AND volume spike
            if price > upper_band and price > ema50 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band AND below EMA50 AND volume spike
            elif price < lower_band and price < ema50 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below EMA50 OR below lower Donchian band
            if price < ema50 or price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above EMA50 OR above upper Donchian band
            if price > ema50 or price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals