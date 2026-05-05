#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation
# Long when price breaks above Donchian(20) high AND price > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Donchian(20) low AND price < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price crosses Donchian(20) midpoint OR price crosses EMA34(1d) in opposite direction
# Donchian channels provide clear structural breakouts with defined risk
# 1d EMA34 ensures alignment with higher timeframe trend to avoid whipsaws
# Volume spike confirms institutional participation and reduces false breakouts
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 12h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    if len(high) >= 20 and len(low) >= 20:
        # Donchian high: highest high over 20 periods
        dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian low: lowest low over 20 periods
        dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Donchian midpoint: average of high and low
        dc_mid = (dc_high + dc_low) / 2.0
    else:
        dc_high = np.full(n, np.nan)
        dc_low = np.full(n, np.nan)
        dc_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or 
            np.isnan(dc_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND price > EMA34(1d) AND volume spike
            if (close[i] > dc_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND price < EMA34(1d) AND volume spike
            elif (close[i] < dc_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR price < EMA34(1d) (trend flip)
            if (close[i] < dc_mid[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR price > EMA34(1d) (trend flip)
            if (close[i] > dc_mid[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals