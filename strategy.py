#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA200 trend filter.
# Uses 4h for direction (breakout of 20-period high/low) and 1d EMA200 for long-term trend filter.
# Volume confirmation on breakout reduces false signals. Session filter (08-20 UTC) avoids low-liquidity hours.
# Target: 15-30 trades/year per symbol (60-120 over 4 years) to minimize fee drag.
# Works in bull/bear markets: trend filter prevents counter-trend trades, volume confirms breakout strength.
name = "1h_Donchian20_4hBreakout_Volume_EMA200"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4-period Donchian channels (20 periods = 80 bars of 4h, but we use 20 for responsiveness)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 1h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        ema_200_val = ema_200_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high, above 1d EMA200, volume confirmation
            if price > donchian_high and price > ema_200_val and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian low, below 1d EMA200, volume confirmation
            elif price < donchian_low and price < ema_200_val and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low or below 1d EMA200
            if price < donchian_low or price < ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high or above 1d EMA200
            if price > donchian_high or price > ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals