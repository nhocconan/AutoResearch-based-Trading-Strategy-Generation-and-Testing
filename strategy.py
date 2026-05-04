#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume confirmation
# In trending markets (1d ADX > 25): breakout in trend direction
# In ranging markets (1d ADX <= 25): fade Donchian touches (mean reversion)
# Volume confirmation (>1.3x 20-period EMA) filters low-quality breakouts
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.

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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) with proper min_periods
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
    
    # Align 1d ADX to 4h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = highest_high.values
    donchian_low = lowest_low.values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_up[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            if adx_aligned[i] > 25:
                # Trending market: breakout in trend direction
                # Determine trend direction using 1d +DI/-DI
                plus_di_14 = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
                minus_di_14 = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
                plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di_14.values)
                minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di_14.values)
                
                if plus_di_aligned[i] > minus_di_aligned[i]:
                    # Uptrend: long on break above Donchian upper
                    if close[i] > donchian_up[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                else:
                    # Downtrend: short on break below Donchian lower
                    if close[i] < donchian_low[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: fade Donchian touches (mean reversion)
                if close[i] <= donchian_low[i] and volume_confirm:
                    # Long at lower band
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_up[i] and volume_confirm:
                    # Short at upper band
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint OR ADX weakens (<20) OR volume drops
            midpoint = (donchian_up[i] + donchian_low[i]) / 2
            if (close[i] >= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR ADX weakens (<20) OR volume drops
            midpoint = (donchian_up[i] + donchian_low[i]) / 2
            if (close[i] <= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals