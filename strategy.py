#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
    # Long when price breaks above 20-period Donchian high AND 1d ATR ratio < 0.8 (low volatility regime) AND volume > 1.2x 20-period MA.
    # Short when price breaks below 20-period Donchian low AND same filters.
    # Exit when price crosses 20-period Donchian midpoint.
    # Uses discrete position sizing (0.25) to target 50-150 trades over 4 years.
    # Works in bull/bear via volatility regime filter avoiding false breakouts in choppy markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio: current ATR / 20-period ATR MA (to detect volatility regime)
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma_20 > 0, atr_1d / atr_ma_20, 1.0)
    
    # Calculate 1d volume 20-period MA
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Calculate 4h Donchian channels (20-period)
    hh_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    ll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (hh_20 + ll_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(hh_20[i]) or np.isnan(ll_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period average
        volume_spike = volume_1d_aligned[i] > 1.2 * vol_ma_1d_aligned[i]
        
        # Volatility regime filter: only trade in low volatility (atr_ratio < 0.8)
        vol_filter = atr_ratio_aligned[i] < 0.8
        
        # Donchian breakout conditions
        breakout_long = close[i] > hh_20[i-1]  # Break above previous period high
        breakout_short = close[i] < ll_20[i-1]  # Break below previous period low
        
        # Exit conditions: price crosses Donchian midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        # Entry conditions
        if breakout_long and volume_spike and vol_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and volume_spike and vol_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_atr_vol_filter_v1"
timeframe = "4h"
leverage = 1.0