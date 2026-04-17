# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout with 1d volume spike and ADX trend filter
# Breakouts capture momentum in trending markets.
# Volume surge confirms institutional interest.
# ADX > 25 filters chop, ensuring trades only in trending conditions.
# Works in bull/bear by trading breakouts in direction of 1d trend.
# Position size: 0.25 for balanced risk/return.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for volume, ADX, and Donchian bands ===
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX calculation (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d volume and its 20-period average
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    
    # 4h Donchian breakout signals
    donchian_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or \
           np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 20-period average volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > volume_ma20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        # Combined filter
        filter_ok = volume_filter and trend_filter
        
        # Determine 1d trend direction using price vs 20-period EMA
        ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
        ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
        trend_up = close_1d[-1] > ema_20_1d_aligned[i] if len(close_1d) > 0 else False
        
        if position == 0:
            # Long when price breaks above 4h Donchian high AND 1d trend is up
            if (close[i] > donchian_high_4h[i] and 
                trend_up and filter_ok):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 4h Donchian low AND 1d trend is down
            elif (close[i] < donchian_low_4h[i] and 
                  not trend_up and filter_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low or filter fails
            if (close[i] < donchian_low_4h[i] or 
                not filter_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high or filter fails
            if (close[i] > donchian_high_4h[i] or 
                not filter_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolume_ADX_TrendFilter"
timeframe = "4h"
leverage = 1.0