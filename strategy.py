#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period average
# Short when price breaks below Donchian(20) low AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period average
# Exit when price reverses to Donchian(10) midpoint OR trend flips (price crosses 1d EMA34)
# Uses discrete sizing (0.30) to balance risk and return. Target: 20-50 trades/year per symbol.
# Donchian channels provide clear breakout levels, 1d EMA34 filters counter-trend trades, volume confirms conviction.

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
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
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels on 4h data
    # Donchian(20) for breakout signals
    if len(high) >= 20:
        donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high_20 = np.full(n, np.nan)
        donchian_low_20 = np.full(n, np.nan)
    
    # Donchian(10) for exit signals (midpoint)
    if len(high) >= 10:
        donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
        donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
        donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
    else:
        donchian_mid_10 = np.full(n, np.nan)
    
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
            np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_10[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian(20) high AND price > 1d EMA34 AND volume spike
            if (close[i] > donchian_high_20[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price < Donchian(20) low AND price < 1d EMA34 AND volume spike
            elif (close[i] < donchian_low_20[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price < Donchian(10) midpoint OR price < 1d EMA34 (trend flip)
            if (close[i] < donchian_mid_10[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price > Donchian(10) midpoint OR price > 1d EMA34 (trend flip)
            if (close[i] > donchian_mid_10[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals