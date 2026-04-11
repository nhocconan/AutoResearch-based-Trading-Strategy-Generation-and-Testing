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
    
    # 1d OHLC for Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period) - using previous day's data
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 1d bars
    donchian_high_20 = np.roll(donchian_high_20, 1)
    donchian_low_20 = np.roll(donchian_low_20, 1)
    donchian_high_20[0] = np.nan
    donchian_low_20[0] = np.nan
    
    # Align 1d Donchian channels to 4h timeframe
    dh_4h = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    dl_4h = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ADX for trend strength (14 period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_dm = tr[1:]
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(dh_4h[i]) or np.isnan(dl_4h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (2.0x average)
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Trend filter: ADX > 25 (strong trend)
        trend_filter = adx[i] > 25
        
        # Long conditions: price breaks above Donchian high with volume and trend
        long_signal = volume_confirmed and trend_filter and (price_high > dh_4h[i])
        
        # Short conditions: price breaks below Donchian low with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < dl_4h[i])
        
        # Exit when price returns to the opposite side of the Donchian channel
        exit_long = position == 1 and price_close < dl_4h[i]
        exit_short = position == -1 and price_close > dh_4h[i]
        
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

# Hypothesis: Donchian breakout strategy with volume and ADX filters for 4h timeframe.
# Enters long when 4h price breaks above 20-day Donchian high with volume >2.0x average and ADX>25.
# Enters short when price breaks below 20-day Donchian low with same conditions.
# Exits when price returns to the opposite side of the Donchian channel.
# ADX filter ensures trades only in strong trending markets, reducing false breakouts in ranging conditions.
# Target: 20-30 trades per year to minimize fee drift while capturing strong trends.