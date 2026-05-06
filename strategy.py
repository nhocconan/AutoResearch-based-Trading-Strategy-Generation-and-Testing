#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and 1d EMA34 trend filter
# - Long when price breaks above 1d Donchian high (20 periods) with volume expansion and price above 1d EMA34
# - Short when price breaks below 1d Donchian low (20 periods) with volume expansion and price below 1d EMA34
# - Exit when price crosses back below/above 1d EMA34
# - Volume filter requires current volume > 1.5x 20-period average to ensure strong breakouts
# - Designed to capture strong trends while avoiding whipsaws in ranging markets
# - Target: 40-80 total trades over 4 years (10-20/year) with 0.30 position sizing
# - Uses higher timeframe (1d) structure for better signal quality and lower trade frequency

name = "4h_DonchianBreakout_1dEMA34_Volume"
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
    
    # Get 1d data for Donchian and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: rolling max of high over 20 periods
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 periods
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation - stricter threshold
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume expansion and above EMA34
            if close[i] > donchian_high_4h[i] and volume_expansion[i] and close[i] > ema_34_1d_4h[i]:
                signals[i] = 0.30
                position = 1
            # Short breakout: price breaks below Donchian low with volume expansion and below EMA34
            elif close[i] < donchian_low_4h[i] and volume_expansion[i] and close[i] < ema_34_1d_4h[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals