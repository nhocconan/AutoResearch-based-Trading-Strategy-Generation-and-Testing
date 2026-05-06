#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and 12h EMA50 trend filter
# - Long when price breaks above 12h Donchian high (20 periods) with volume expansion and price above 12h EMA50
# - Short when price breaks below 12h Donchian low (20 periods) with volume expansion and price below 12h EMA50
# - Exit when price crosses back below/above 12h EMA50
# - Volume filter requires current volume > 1.3x 20-period average
# - Designed to capture strong trends while avoiding whipsaws in ranging markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_DonchianBreakout_12hEMA50_Volume"
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
    
    # Get 12h data for Donchian and EMA calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high: rolling max of high over 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 periods
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)  # Volume confirmation
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(ema_50_12h_4h[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume expansion and above EMA50
            if close[i] > donchian_high_4h[i] and volume_expansion[i] and close[i] > ema_50_12h_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low with volume expansion and below EMA50
            elif close[i] < donchian_low_4h[i] and volume_expansion[i] and close[i] < ema_50_12h_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA50
            if close[i] < ema_50_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA50
            if close[i] > ema_50_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals