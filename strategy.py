#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Donchian upper channel AND price > EMA34(1d) AND volume > 2.0x 24-period average
# Short when price breaks below Donchian lower channel AND price < EMA34(1d) AND volume > 2.0x 24-period average
# Exit when price crosses back below Donchian middle (long exit) or above Donchian middle (short exit)
# Donchian channels provide clear structure, 1d EMA34 avoids counter-trend trades in bear markets, volume confirms conviction
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
    
    # Calculate Donchian channels (20-period) on 12h data
    if len(high) >= 20 and len(low) >= 20:
        upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        middle = (upper + lower) / 2.0
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 24-period average (spike filter)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (2.0 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(middle[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND price > EMA34(1d) AND volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND price < EMA34(1d) AND volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Donchian middle (mean reversion)
            if close[i] < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Donchian middle (mean reversion)
            if close[i] > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals