#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper channel AND price > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below 12h Donchian lower channel AND price < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price returns to Donchian midpoint (mean reversion) OR trend flips (price crosses 1d EMA50)
# Donchian channels provide clear structural breakouts, EMA50(1d) filters counter-trend whipsaws, volume confirms conviction
# Target: 12-37 trades/year per symbol (50-150 total over 4 years)
# Discrete sizing (0.25) to limit fee drag, works in both bull (breakouts) and bear (mean reversion in ranges)

name = "12h_Donchian20_VolumeSpike_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 12h timeframe (20-period)
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    middle_channel = np.full(n, np.nan)
    
    if len(high) >= 20 and len(low) >= 20:
        # Calculate rolling max/min for Donchian channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        upper_channel = high_series.rolling(window=20, min_periods=20).max().values
        lower_channel = low_series.rolling(window=20, min_periods=20).min().values
        middle_channel = (upper_channel + lower_channel) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper channel AND price > EMA50(1d) AND volume spike
            if (close[i] > upper_channel[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower channel AND price < EMA50(1d) AND volume spike
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint (mean reversion) OR price < EMA50(1d) (trend flip)
            if (close[i] <= middle_channel[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint (mean reversion) OR price > EMA50(1d) (trend flip)
            if (close[i] >= middle_channel[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals