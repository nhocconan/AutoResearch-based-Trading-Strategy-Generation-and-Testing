#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d volume spike + choppiness regime filter
    # Long when: price breaks above Donchian(20) high AND volume > 2.0x avg volume AND chop > 61.8 (range)
    # Short when: price breaks below Donchian(20) low AND volume > 2.0x avg volume AND chop > 61.8 (range)
    # Exit when: price crosses Donchian midpoint OR chop < 38.2 (trend) OR volume drops below average
    # Uses discrete sizing (0.25) targeting 75-200 total trades over 4 years.
    # Works in bull/bear via chop filter avoiding false breakouts in strong trends.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for choppiness index
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate choppiness index: CHOP = 100 * log10(sum(ATR14) / (max_high - min_low)) / log10(14)
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (hh - ll + 1e-10)) / np.log10(14)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian(20) channels on 4h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_threshold[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation (both 4h and 1d)
        vol_ok_4h = volume[i] > vol_threshold[i]
        vol_ok_1d = volume[i] > vol_ma_1d_aligned[i] * 1.5  # 1.5x 1d avg volume
        
        # Chop regime: range market (chop > 61.8) for mean reversion
        chop_ok = chop[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_ok_4h and vol_ok_1d and chop_ok and position != 1
        short_entry = short_breakout and vol_ok_4h and vol_ok_1d and chop_ok and position != -1
        
        # Exit conditions: price crosses Donchian midpoint OR chop < 38.2 (trend) OR volume drops below average
        exit_long = close[i] < donchian_mid[i] or chop[i] < 38.2 or volume[i] < vol_ma[i]
        exit_short = close[i] > donchian_mid[i] or chop[i] < 38.2 or volume[i] < vol_ma[i]
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
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

name = "4h_1d_donchian_chop_volume_v2"
timeframe = "4h"
leverage = 1.0