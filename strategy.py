#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams %R mean reversion with 1-week ADX trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# Weekly ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Volume > 1.3x average confirms institutional participation.
# Exits occur when Williams %R returns to neutral territory (-50) or opposite extreme.
# Designed for 12-hour timeframe targeting 20-30 trades per year per symbol (80-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX on weekly data
    adx_len = 14
    if len(df_1w) < adx_len * 2:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = pd.Series(df_1w['high']).diff()
    minus_dm = pd.Series(df_1w['low'].shift()) - pd.Series(df_1w['low'])
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(alpha=1/adx_len, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/adx_len, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/adx_len, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/adx_len, adjust=False).mean()
    
    adx_values = adx.values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # Williams %R on 12-hour data (using current timeframe data)
    willr_len = 14
    highest_high = pd.Series(high).rolling(window=willr_len, min_periods=willr_len).max()
    lowest_low = pd.Series(low).rolling(window=willr_len, min_periods=willr_len).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr_values = willr.values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, willr_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(willr_values[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trending = adx_1w_aligned[i] > 25
        
        # Williams %R levels
        oversold = willr_values[i] < -80
        overbought = willr_values[i] > -20
        neutral = abs(willr_values[i] + 50) < 10  # Near -50
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: oversold + trending + volume
            if oversold and trending and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: overbought + trending + volume
            elif overbought and trending and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: returns to neutral or becomes overbought
            if neutral or overbought:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: returns to neutral or becomes oversold
            if neutral or oversold:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_WilliamsR_ADX_Volume_v1"
timeframe = "12h"
leverage = 1.0