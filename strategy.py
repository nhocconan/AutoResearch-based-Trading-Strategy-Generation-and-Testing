#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation + choppiness regime filter
# Long when price breaks above Donchian upper(20) AND close > EMA34(1d) AND volume > 1.5x 20-period average AND chop < 61.8 (trending regime)
# Short when price breaks below Donchian lower(20) AND close < EMA34(1d) AND volume > 1.5x 20-period average AND chop < 61.8 (trending regime)
# Exit when price retraces to Donchian midpoint OR EMA34(1d) trend flip
# Uses 4h primary timeframe with 1d HTF for trend filter and chop regime to avoid whipsaw and reduce overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag
# Donchian channels provide structural breakouts; volume and trend filter confirm strength; chop filter avoids ranging markets

name = "4h_Donchian20_Breakout_1dEMA34_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate choppiness index on 1d for regime filter
    if len(df_1d) >= 14:
        # True Range
        tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
        tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
        tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: CHOP = 100 * log10(sum(TR14)/(HH14-LL14)) / log10(14)
        sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        chop_1d = 100 * (np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14))
        # Handle division by zero or invalid values
        chop_1d = np.where((hh_14 - ll_14) > 0, chop_1d, 50.0)  # Default to neutral when range is zero
        chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)
    else:
        chop_1d = np.full(len(df_1d), 50.0)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian channels on 4h (20-period)
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average (moderate to balance trades and confirmation)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND close > EMA34(1d) AND volume spike AND trending regime (chop < 61.8)
            if (high[i] > donchian_upper[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i] and 
                chop_1d_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND close < EMA34(1d) AND volume spike AND trending regime (chop < 61.8)
            elif (low[i] < donchian_lower[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i] and 
                  chop_1d_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Donchian midpoint OR close < EMA34(1d) (trend flip)
            if close[i] <= donchian_mid[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Donchian midpoint OR close > EMA34(1d) (trend flip)
            if close[i] >= donchian_mid[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals