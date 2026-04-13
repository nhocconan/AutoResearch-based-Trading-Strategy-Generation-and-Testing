#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ADX regime filter
    # Long: price breaks above H3 + 4h volume > 1.5x 20-period average + 1d ADX > 25
    # Short: price breaks below L3 + 4h volume > 1.5x 20-period average + 1d ADX > 25
    # Exit: price returns to Camarilla pivot point (PP)
    # Uses Camarilla pivots for structure (works in ranging markets), volume for confirmation, ADX for trend filter
    # Designed for 1h timeframe with 4h/1d HTF to reduce noise and overtrading
    # Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary HTF analysis
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for volume and ADX (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivots on 4h data (using previous bar's OHLC)
    # Camarilla levels: PP = (H+L+C)/3, H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # We use shifted values to avoid look-ahead (previous bar's OHLC)
    pp_4h = (high_4h[:-1] + low_4h[:-1] + close_4h[:-1]) / 3.0
    range_4h = high_4h[:-1] - low_4h[:-1]
    h3_4h = close_4h[:-1] + range_4h * 1.1 / 4.0
    l3_4h = close_4h[:-1] - range_4h * 1.1 / 4.0
    
    # Prepend NaN for first bar (no previous bar)
    pp_4h = np.concatenate([[np.nan], pp_4h])
    h3_4h = np.concatenate([[np.nan], h3_4h])
    l3_4h = np.concatenate([[np.nan], l3_4h])
    
    # Align Camarilla levels to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align volume average and ADX to 1h timeframe
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):  # start from 20 to have enough data for calculations
        # Skip if data not ready or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_avg_20_4h_aligned[i]) or np.isnan(adx_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        curr_vol_4h = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_confirmed = curr_vol_4h > 1.5 * vol_avg_20_4h_aligned[i]
        
        # Regime filter: ADX > 25 (strong trending market)
        trending = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_long = close[i] > h3_aligned[i] and volume_confirmed and trending
        breakout_short = close[i] < l3_aligned[i] and volume_confirmed and trending
        
        # Exit conditions: return to Camarilla pivot point
        exit_long = position == 1 and close[i] <= pp_aligned[i]
        exit_short = position == -1 and close[i] >= pp_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "1h"
leverage = 1.0