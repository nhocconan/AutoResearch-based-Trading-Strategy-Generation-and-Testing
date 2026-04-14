#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 1d ADX trend filter and volume confirmation
# Choppiness Index > 61.8 indicates ranging market (mean revert), < 38.2 indicates trending (trend follow)
# 1d ADX > 25 confirms trend strength for trend-following entries
# Volume > 1.5x average confirms institutional participation
# Works in bull/bear by adapting to market regime
# Target: 20-40 trades/year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX(14) for trend strength
    adx_len = 14
    if len(df_1d) < adx_len:
        return np.zeros(n)
    
    # Calculate ADX components
    plus_dm = np.diff(df_1d['high'], prepend=df_1d['high'].iloc[0])
    minus_dm = np.diff(df_1d['low'], prepend=df_1d['low'].iloc[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(df_1d['high']))
    tr2 = np.abs(np.diff(df_1d['low']))
    tr3 = np.abs(df_1d['close'].shift(1) - df_1d['high'])
    tr4 = np.abs(df_1d['close'].shift(1) - df_1d['low'])
    tr = np.maximum.reduce([tr1, tr2, tr3, tr4])
    tr = np.concatenate([[tr[0]], tr])  # align length
    
    atr = pd.Series(tr).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    adx_1d = align_htf_to_ltf(prices, df_1d, adx)
    
    # Choppiness Index (14) on 1d
    chop_len = 14
    if len(df_1d) < chop_len:
        return np.zeros(n)
    
    atr_sum = pd.Series(tr).rolling(window=chop_len, min_periods=chop_len).sum().values
    hh = df_1d['high'].rolling(window=chop_len, min_periods=chop_len).max().values
    ll = df_1d['low'].rolling(window=chop_len, min_periods=chop_len).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(chop_len)
    chop_1d = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian channel (20 periods)
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(adx_1d[i]) or
            np.isnan(chop_1d[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Choppiness Index
        ranging = chop_1d[i] > 61.8  # mean revert regime
        trending = chop_1d[i] < 38.2  # trend follow regime
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d[i] > 25
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: trending market + ADX strength + Donchian breakout + volume
            if (trending and strong_trend and 
                close[i] > dc_upper[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: trending market + ADX strength + Donchian breakdown + volume
            elif (trending and strong_trend and 
                  close[i] < dc_lower[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            # Enter mean reversion: ranging market + Donchian reversal + volume
            elif (ranging and 
                  volume_confirmed):
                if close[i] < dc_lower[i]:  # oversold - go long
                    position = 1
                    signals[i] = position_size
                elif close[i] > dc_upper[i]:  # overbought - go short
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakness or mean reversion signal
            if (not trending or not strong_trend or 
                chop_1d[i] > 50 or  # exiting trending regime
                close[i] < dc_lower[i]):  # stop loss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakness or mean reversion signal
            if (not trending or not strong_trend or 
                chop_1d[i] > 50 or  # exiting trending regime
                close[i] > dc_upper[i]):  # stop loss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Choppiness_ADX_Volume_v1"
timeframe = "4h"
leverage = 1.0