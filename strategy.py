#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian channel breakout with volume confirmation and ADX trend filter
# Daily Donchian(20) identifies key breakout levels. Breakouts above upper channel or below lower channel
# with volume > 1.5x 20-period average indicate strong momentum. ADX > 25 ensures trend strength.
# Works in bull/bear markets: breakouts capture trends, ADX filter avoids weak/choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "12h_1dDonchian20_Volume_ADXFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channel (20-period) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channel: 20-period high and low
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # ADX trend filter (14-period)
    # Calculate +DI, -DI, and DX
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (tr_ma + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (tr_ma + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_filter = adx > 25
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume and trend confirmation
            if close[i] > upper_20_aligned[i] and volume_filter[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian with volume and trend confirmation
            elif close[i] < lower_20_aligned[i] and volume_filter[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (failed support) or ADX weakens
            if close[i] < lower_20_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (failed resistance) or ADX weakens
            if close[i] > upper_20_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals