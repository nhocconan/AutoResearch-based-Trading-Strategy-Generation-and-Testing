#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA trend filter.
# In bull markets: buy breakouts above 4h Donchian high when 1d EMA50 is rising.
# In bear markets: sell breakdowns below 4h Donchian low when 1d EMA50 is falling.
# Uses volume spike to confirm breakout strength and reduce false signals.
# Target: 15-37 trades/year to avoid fee drag.
name = "1h_DonchianBreakout_4hTrend_1dEMA_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 24-period EMA (1h chart)
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_confirm = volume > (1.5 * vol_ema24)
    
    # Align 4h indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Align 1d EMA to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high + 1d uptrend + volume spike
            if (price > donchian_high_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian low + 1d downtrend + volume spike
            elif (price < donchian_low_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 4h Donchian high or trend reverses
            if price < donchian_high_aligned[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns above 4h Donchian low or trend reverses
            if price > donchian_low_aligned[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals