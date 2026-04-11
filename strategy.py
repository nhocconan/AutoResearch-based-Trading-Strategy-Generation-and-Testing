#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_trend_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 1d bars
    donch_high_1d = np.roll(donch_high_1d, 1)
    donch_low_1d = np.roll(donch_low_1d, 1)
    donch_high_1d[0] = np.nan
    donch_low_1d[0] = np.nan
    
    # Align 1d Donchian levels to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_4h = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # 4h ADX for trend strength filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.8x 20-period average (balance between signal quality and trade count)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Trend filter: ADX > 30 for trending market
        trend_filter = adx_val > 30
        
        # Long conditions: price breaks above 1d Donchian high with volume and trend
        long_signal = volume_confirmed and trend_filter and (price_high > donch_high_4h[i])
        
        # Short conditions: price breaks below 1d Donchian low with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < donch_low_4h[i])
        
        # Exit when price returns to the middle of 1d Donchian channel
        donch_mid_1d = (donch_high_1d + donch_low_1d) / 2
        donch_mid_4h = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
        exit_long = position == 1 and price_close < donch_mid_4h[i]
        exit_short = position == -1 and price_close > donch_mid_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1d Donchian breakout with volume confirmation and ADX trend filter.
# Enters long when 4h price breaks above 20-day 1d Donchian high with volume >1.8x average and ADX>30.
# Enters short when price breaks below 20-day 1d Donchian low with same conditions.
# Exits when price returns to the midpoint of the 1d Donchian channel.
# Works in both bull and breakout markets by capturing strong directional moves.
# Moderate thresholds target ~20-30 trades/year to minimize fee drag.