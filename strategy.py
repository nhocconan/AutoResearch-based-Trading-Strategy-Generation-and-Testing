#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and ATR filter
# Donchian(20) on 1d provides clear trend direction and structure
# Long when price breaks above 1d Donchian upper with volume confirmation
# Short when price breaks below 1d Donchian lower with volume confirmation
# Uses ATR(14) for volatility filtering and discrete position sizing 0.25
# Works in bull/bear markets: breakout follows trends, ATR filter avoids low-volatility whipsaws
# Target: ~15-25 trades/year to minimize fee drag while maintaining edge

name = "12h_1d_donchian_breakout_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper_20 = rolling_max(high_1d, 20)
    donchian_lower_20 = rolling_min(low_1d, 20)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x average 12h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # ATR filter: only trade when volatility is sufficient (ATR > 0.5 * 20-period MA of ATR)
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr_14_aligned[i] > 0.5 * atr_ma_20[i] if not np.isnan(atr_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if price falls below Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Donchian breakout with volume and volatility confirmation
            if (close[i] > donchian_upper_aligned[i] and volume_confirmed and vol_filter):
                position = 1
                signals[i] = 0.25
            elif (close[i] < donchian_lower_aligned[i] and volume_confirmed and vol_filter):
                position = -1
                signals[i] = -0.25
    
    return signals