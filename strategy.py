#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and ADX regime filter
# In trending markets (1d ADX >= 25): breakout above/below Donchian(20) with volume > 1.5x 20-period EMA
# In ranging markets (1d ADX < 25): fade Donchian extremes with volume confirmation
# Uses discrete sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year).
# Works in both bull/bear via regime filter - trend following in strong trends, mean reversion in chop.

name = "12h_Donchian20_1dADX_VolumeRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Donchian channel (20-period) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<25) or trending (ADX>=25)
            if adx_aligned[i] < 25:
                # Ranging market: fade Donchian extremes (mean reversion)
                if close[i] <= lowest_low[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= highest_high[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: Donchian breakout in trend direction
                if close[i] > highest_high[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel OR ADX weakens (<20) OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] <= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel OR ADX weakens (<20) OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] >= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals