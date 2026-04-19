#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout and 1d trend filter.
# In bull markets, price breaks above 4h Donchian high with 1d uptrend -> long.
# In bear markets, price breaks below 4h Donchian low with 1d downtrend -> short.
# Uses volume confirmation to avoid false breaks. Session filter (08-20 UTC) reduces noise.
# Target: 15-37 trades/year (60-150 over 4 years) by requiring confluence of 4h breakout, 1d trend, and volume.
name = "1h_4hDonchian_1dTrend_Volume_Session"
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
    
    # Get 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(high_20_4h_aligned[i]) or np.isnan(low_20_4h_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high + 1d uptrend + volume spike
            if price > high_20_4h_aligned[i] and price > ema_50_1d_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian low + 1d downtrend + volume spike
            elif price < low_20_4h_aligned[i] and price < ema_50_1d_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below 4h Donchian low (reversal signal)
            if price < low_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price breaks above 4h Donchian high (reversal signal)
            if price > high_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals