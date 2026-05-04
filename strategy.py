#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume confirmation
# In ranging markets (1d ADX < 25): Donchian breakout = mean reversion (fade the breakout)
# In trending markets (1d ADX >= 25): Donchian breakout = trend continuation
# Volume confirmation (>1.3x 20-period EMA) ensures participation. Uses discrete sizing (0.25) to minimize fees.
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# BTC/ETH edge: Donchian captures structure; ADX regime avoids whipsaws; volume confirms institutional interest.

name = "4h_Donchian20_1dADX_Regime_Volume"
timeframe = "4h"
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
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
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
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<25) or trending (ADX>=25)
            if adx_aligned[i] < 25:
                # Ranging market: fade the Donchian breakout (mean reversion)
                if close[i] <= lowest_low[i] and volume_confirm:
                    signals[i] = 0.25  # long at lower band
                    position = 1
                elif close[i] >= highest_high[i] and volume_confirm:
                    signals[i] = -0.25  # short at upper band
                    position = -1
            else:
                # Trending market: continue the Donchian breakout (trend following)
                if close[i] >= highest_high[i] and volume_confirm:
                    signals[i] = 0.25  # long on upper breakout
                    position = 1
                elif close[i] <= lowest_low[i] and volume_confirm:
                    signals[i] = -0.25  # short on lower breakout
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel OR ADX weakening (<20) OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] >= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel OR ADX weakening (<20) OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] <= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals