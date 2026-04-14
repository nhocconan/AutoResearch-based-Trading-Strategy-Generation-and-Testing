#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter + 1-day ADX trend filter + 12-hour Donchian(20) breakout.
# Uses daily ADX > 25 to identify trending markets and 12h Choppiness Index > 61.8 to avoid ranging markets.
# Enters on Donchian breakout in the direction of daily ADX trend (ADX slope) with volume confirmation.
# Exits when price returns to the 12-hour EMA(50) or breaks opposite Donchian band.
# Designed for low trade frequency (15-25/year) to minimize fee drag in 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on daily data
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr
    
    # DX and ADX
    dx = 100 * (pd.Series(plus_di) - pd.Series(minus_di)).abs() / (pd.Series(plus_di) + pd.Series(minus_di))
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # ADX slope for trend direction (rising ADX = strengthening trend)
    adx_slope = pd.Series(adx_values).diff(3)  # 3-period slope
    
    # Load 12h data ONCE for Choppiness Index and Donchian
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Choppiness Index(14) on 12h data
    # True Range
    tr1_12h = pd.Series(df_12h['high']).diff().abs()
    tr2_12h = (pd.Series(df_12h['high']) - pd.Series(df_12h['close'].shift())).abs()
    tr3_12h = (pd.Series(df_12h['low']) - pd.Series(df_12h['close'].shift())).abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h_sum = tr_12h.rolling(window=14, min_periods=14).sum()
    
    # Max and min over 14 periods
    max_hh = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max()
    min_ll = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(atr_sum / (max_h - min_l)) / log10(14)
    chop = 100 * np.log10(atr_12h_sum / (max_hh - min_ll)) / np.log10(14)
    chop_values = chop.values
    
    # EMA(50) on 12h for exit
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channel (20 periods) on 12h
    dc_len = 20
    dc_upper = pd.Series(df_12h['high']).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(df_12h['low']).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume on 12h
    vol_ma = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    adx_slope_aligned = align_htf_to_ltf(prices, df_1d, adx_slope.values)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_values)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    dc_upper_aligned = align_htf_to_ltf(prices, df_12h, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_12h, dc_lower)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper_aligned[i]) or 
            np.isnan(dc_lower_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(adx_slope_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25  # ADX > 25 = trending market
        not_choppy = chop_aligned[i] <= 61.8  # Chop <= 61.8 = not ranging
        adx_rising = adx_slope_aligned[i] > 0  # ADX slope positive = strengthening trend
        
        # Volume confirmation
        volume_confirmed = df_12h['volume'].iloc[i] > 1.5 * vol_ma_aligned[i] if i < len(df_12h) else volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + ADX rising + not choppy + volume
            if (i < len(df_12h) and 
                df_12h['close'].iloc[i] > dc_upper_aligned[i] and 
                adx_rising and 
                trending and 
                not_choppy and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + ADX rising + not choppy + volume
            elif (i < len(df_12h) and 
                  df_12h['close'].iloc[i] < dc_lower_aligned[i] and 
                  adx_rising and 
                  trending and 
                  not_choppy and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 12h EMA or breaks below Donchian lower
            if (i < len(df_12h) and 
                (df_12h['close'].iloc[i] < ema_12h_aligned[i] or 
                 df_12h['close'].iloc[i] < dc_lower_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 12h EMA or breaks above Donchian upper
            if (i < len(df_12h) and 
                (df_12h['close'].iloc[i] > ema_12h_aligned[i] or 
                 df_12h['close'].iloc[i] > dc_upper_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_ADX_Chop_Donchian_Volume_v1"
timeframe = "12h"
leverage = 1.0