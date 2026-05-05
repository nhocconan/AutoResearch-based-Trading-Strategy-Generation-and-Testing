#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 6h EMA50 trend filter
# Long when: price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period average AND 6h close > 6h EMA50
# Short when: price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period average AND 6h close < 6h EMA50
# Exit when price returns to Donchian(20) midpoint (mean reversion in channel)
# Donchian channels provide clear breakout levels with built-in stoploss
# Volume spike confirms institutional participation
# 6h EMA50 filter ensures alignment with intermediate trend
# Target: 100-200 total trades over 4 years (25-50/year) with discrete sizing 0.30 to balance profit and fees

name = "4h_Donchian20_1dVolumeSpike_6hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Get 6h data ONCE before loop for EMA50 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 1d average volume (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h EMA(50) for trend filter
    ema_50_6h = pd.Series(close_6h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_6h, ema_50_6h)
    
    # Calculate Donchian Channel (20) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high + volume spike + 6h uptrend
            if (close[i] > donchian_high[i] and 
                volume[i] > 2.0 * vol_ma_aligned[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: break below Donchian low + volume spike + 6h downtrend
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 2.0 * vol_ma_aligned[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: return to Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: return to Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals