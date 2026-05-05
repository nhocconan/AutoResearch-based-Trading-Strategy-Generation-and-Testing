#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter (EMA50) and volume confirmation
# Long when price breaks above Donchian(20) high AND 12h EMA50 trending up AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND 12h EMA50 trending down AND volume > 1.5x 20-period average
# Exit when price crosses Donchian(20) midpoint OR volume drops below average
# Donchian channels provide clear breakout levels with built-in volatility adjustment
# 12h EMA50 filters for higher timeframe trend alignment to avoid counter-trend whipsaws
# Volume confirmation ensures breakouts have institutional participation
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_Donchian20_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels: upper = max(high,20), lower = min(low,20)
    if len(high) >= 20:
        rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (rolling_max + rolling_min) / 2.0
    else:
        rolling_max = np.full(n, np.nan)
        rolling_min = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
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
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(rolling_max[i]) or 
            np.isnan(rolling_min[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 12h EMA50 trending up AND volume spike
            if (close[i] > rolling_max[i] and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[max(0, i-1)] and  # EMA50 trending up
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 12h EMA50 trending down AND volume spike
            elif (close[i] < rolling_min[i] and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[max(0, i-1)] and  # EMA50 trending down
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR volume drops below average
            if (close[i] < donchian_mid[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR volume drops below average
            if (close[i] > donchian_mid[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals