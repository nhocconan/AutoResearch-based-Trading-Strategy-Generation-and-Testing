#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume spike filter and 1d chop regime.
    # Long when price breaks above Donchian(20) high + volume > 2x 20-period average + CHOP(14) > 61.8 (range).
    # Short when price breaks below Donchian(20) low + volume > 2x 20-period average + CHOP(14) > 61.8 (range).
    # Exit when price crosses Donchian(20) midpoint.
    # Uses breakout structure with volume confirmation and range regime to avoid false breakouts in trends.
    # Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 12h data for volume spike (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate volume average (20-period) on 12h
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF volume average to 4h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Get 1d data for chop regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR(14) on 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods on 1d
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Chopiness Index (CHOP) on 1d
    chop_denom = atr_1d * 14
    chop_num = hh_1d - ll_1d
    chop = np.where(chop_denom != 0, 100 * np.log10(chop_num / chop_denom) / np.log10(14), 50)
    
    # Align HTF chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2x 20-period average
        # Need to get the 12h volume aligned to 4h - get the corresponding 12h volume bar
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirm = vol_12h_aligned[i] > 2.0 * vol_ma_12h_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion/breakout fade)
        regime_filter = chop_aligned[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry conditions: breakout + volume + regime
        long_signal = long_breakout and volume_confirm and regime_filter
        short_signal = short_breakout and volume_confirm and regime_filter
        
        # Exit conditions: price crosses Donchian midpoint
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "4h_12h_1d_donchian_vol_chop_breakout_v1"
timeframe = "4h"
leverage = 1.0