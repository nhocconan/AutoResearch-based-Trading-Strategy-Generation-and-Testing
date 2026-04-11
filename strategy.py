#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20 period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 12h
    dh_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    dl_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 12h ADX for trend filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    tr_s = pd.Series(tr)
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_s.rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_s.rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(dh_12h[i]) or np.isnan(dl_12h[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_high = high[i]
        price_low = low[i]
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: ADX > 25
        trend_filter = adx[i] > 25
        
        # Long conditions: price breaks above daily Donchian high
        long_signal = volume_confirmed and trend_filter and (price_high > dh_12h[i])
        
        # Short conditions: price breaks below daily Donchian low
        short_signal = volume_confirmed and trend_filter and (price_low < dl_12h[i])
        
        # Exit when price crosses the 50% level of the daily Donchian channel
        mid_point = (dh_12h[i] + dl_12h[i]) / 2
        exit_long = position == 1 and price_close < mid_point
        exit_short = position == -1 and price_close > mid_point
        
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

# Hypothesis: Daily Donchian channel breakout strategy for 12h timeframe with volume confirmation (>1.5x average volume) and ADX filter (>25).
# Enters long when 12h price breaks above the daily Donchian high (20-period high) with volume >1.5x average and ADX>25.
# Enters short when price breaks below the daily Donchian low (20-period low) with same conditions.
# Exits when price crosses the midpoint of the daily Donchian channel (mean reversion within the channel).
# Uses volume confirmation and ADX filter to reduce false breakouts and focus on strong trends.
# Target: 15-30 trades per year to minimize fee drift while capturing strong daily trends.
# Daily Donchian channels adapt to volatility and work in both bull and bear markets.