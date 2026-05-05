#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA200 trend filter and 1d volume spike confirmation
# Long when price breaks above Donchian(20) upper band AND price > 1w EMA200 AND 1d volume > 2.0x 20-period average
# Short when price breaks below Donchian(20) lower band AND price < 1w EMA200 AND 1d volume > 2.0x 20-period average
# Exit when price crosses back below/above Donchian(20) mid-band OR 1w trend flips (price crosses EMA200)
# Donchian(20) provides clear structure with defined breakout levels
# 1w EMA200 filters for higher timeframe trend to avoid counter-trend whipsaws in bear markets
# 1d volume spike confirms institutional participation and reduces false breakouts
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_Donchian20_1wEMA200_Trend_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA200 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 1d data ONCE before loop for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d volume and its 20-period average for spike filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align 1d volume spike to 6h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate Donchian(20) on 6h data
    if len(high) >= 20:
        dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        dc_mid = (dc_upper + dc_lower) / 2.0
    else:
        dc_upper = np.full(n, np.nan)
        dc_lower = np.full(n, np.nan)
        dc_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(dc_mid[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND price > 1w EMA200 AND 1d volume spike
            if (close[i] > dc_upper[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND price < 1w EMA200 AND 1d volume spike
            elif (close[i] < dc_lower[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Donchian mid (mean reversion) OR price < 1w EMA200 (trend flip)
            if (close[i] < dc_mid[i] or 
                close[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Donchian mid (mean reversion) OR price > 1w EMA200 (trend flip)
            if (close[i] > dc_mid[i] or 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals