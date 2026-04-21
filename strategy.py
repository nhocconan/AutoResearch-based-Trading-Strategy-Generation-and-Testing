#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and ADX trend filter.
Buy when price breaks above 12h Donchian upper band (20) in strong trend (ADX>25) with volume spike.
Sell when price breaks below 12h Donchian lower band (20) in strong trend with volume spike.
Exit when ADX weakens (<20) or opposite breakout occurs.
12h Donchian filters noise, volume confirms breakout strength, ADX ensures trending market.
Designed for 4h to target 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Load 12h data ONCE before loop for ADX
    # Calculate ADX components: +DI, -DI, DX
    period = 14
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    
    # True Range
    tr1 = high_12h_series - low_12h_series
    tr2 = abs(high_12h_series - close_12h_series.shift(1))
    tr3 = abs(low_12h_series - close_12h_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean().values
    
    # Directional Movement
    up_move = high_12h_series - high_12h_series.shift(1)
    down_move = low_12h_series.shift(1) - low_12h_series
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # 4h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Trend and volume filters
        strong_trend = adx_val > 25
        weak_trend = adx_val < 20
        volume_spike = vol_ratio_val > 1.5
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian high + strong trend + volume spike
            if (price_close > donch_high_val and 
                strong_trend and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian low + strong trend + volume spike
            elif (price_close < donch_low_val and 
                  strong_trend and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: trend weakening or opposite breakout
            if position == 1:
                # Exit long: trend weakens OR price breaks below Donchian low
                if weak_trend or price_close < donch_low_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend weakens OR price breaks above Donchian high
                if weak_trend or price_close > donch_high_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_12hADX_Volume"
timeframe = "4h"
leverage = 1.0