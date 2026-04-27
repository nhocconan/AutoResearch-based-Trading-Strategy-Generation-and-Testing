#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Uses 1d Camarilla pivot levels (R1/S1) for breakout entries with 1d EMA34 trend filter and volume confirmation. 
Added regime filter using 1d ADX to avoid choppy markets. Long when price breaks above R1 AND 1d close > EMA34 (uptrend) 
AND volume > 2.0 * 20-period average AND 1d ADX > 25 (trending). Short when price breaks below S1 AND 1d close < EMA34 
(downtrend) AND volume > 2.0 * 20-period average AND 1d ADX > 25 (trending). Exit when price returns to the pivot level 
(R1 for longs, S1 for shorts) OR trend reverses. Designed for 12h timeframe to achieve 50-150 total trades over 4 years 
with low fee drag. Works in both bull and bear markets by following 1d trend while using Camarilla levels for precise 
breakout entries and avoiding false signals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels: R1, S1
    # Camarilla formulas: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d ADX for regime filter (avoid choppy markets)
    high_1d_series = pd.Series(df_1d['high'].values)
    low_1d_series = pd.Series(df_1d['low'].values)
    close_1d_series = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d_series.diff()
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_smooth = atr_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20), ADX (14+14=28)
    start_idx = max(34, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_conf = volume_confirm[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 with 1d trend filter AND volume AND ADX > 25 (trending)
            # Long: price breaks above R1 (minor resistance) AND 1d uptrend AND volume AND ADX > 25
            long_condition = (close_val > r1_level) and (close_val > ema_val) and vol_conf and (adx_val > 25)
            # Short: price breaks below S1 (minor support) AND 1d downtrend AND volume AND ADX > 25
            short_condition = (close_val < s1_level) and (close_val < ema_val) and vol_conf and (adx_val > 25)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to R1 level OR trend breaks OR ADX drops below 20 (choppy)
            exit_condition = (close_val <= r1_level) or (close_val < ema_val) or (adx_val < 20)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to S1 level OR trend breaks OR ADX drops below 20 (choppy)
            exit_condition = (close_val >= s1_level) or (close_val > ema_val) or (adx_val < 20)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0