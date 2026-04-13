#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1w HTF - Weekly Camarilla pivot breaks with 1d volume confirmation and ADX trend filter
    # Designed to capture institutional breakouts from key weekly pivot levels with volume confirmation
    # Target: 50-150 total trades over 4 years (12-37/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for HTF Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for volume confirmation and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    camarilla_H5 = (high_1w + low_1w + close_1w) / 3 + 1.1 * (high_1w - low_1w) / 2  # R4
    camarilla_H4 = (high_1w + low_1w + close_1w) / 3 + 1.1 * (high_1w - low_1w) / 4  # R3
    camarilla_H3 = (high_1w + low_1w + close_1w) / 3 + 1.1 * (high_1w - low_1w) / 6  # S3
    camarilla_L3 = (high_1w + low_1w + close_1w) / 3 - 1.1 * (high_1w - low_1w) / 6  # S3
    camarilla_L4 = (high_1w + low_1w + close_1w) / 3 - 1.1 * (high_1w - low_1w) / 4  # R3
    camarilla_L5 = (high_1w + low_1w + close_1w) / 3 - 1.1 * (high_1w - low_1w) / 2  # R4
    
    # Calculate 1d ADX (14-period) for trend filter
    def calculate_adx(high, low, close, window=14):
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=window, min_periods=window).sum() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=window, min_periods=window).sum() / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=window, min_periods=window).mean()
        return adx.values
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, window=14)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_H5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H5)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L3)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L4)
    camarilla_L5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L5)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_H5_aligned[i]) or 
            np.isnan(camarilla_H4_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(camarilla_L5_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 1d average volume (scaled)
        # Scale 1d volume to 6h by dividing by 4 (since 1d = 4x 6h)
        volume_confirmed = volume[i] > 1.3 * (vol_avg_20_aligned[i] / 4)
        
        # Trend filter: ADX > 25 for trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        # Breakout conditions at Camarilla levels
        breakout_H4 = close[i] > camarilla_H4_aligned[i]  # Break above R3
        breakdown_L4 = close[i] < camarilla_L4_aligned[i]  # Break below S3
        
        # Entry conditions
        enter_long = breakout_H4 and volume_confirmed and trend_filter
        enter_short = breakdown_L4 and volume_confirmed and trend_filter
        
        # Exit conditions: return to opposite Camarilla level or midpoint
        exit_long = position == 1 and close[i] < camarilla_H3_aligned[i]  # Return below R3
        exit_short = position == -1 and close[i] > camarilla_L3_aligned[i]  # Return above S3
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
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

name = "6h_1w_camarilla_breakout_volume_adx_v1"
timeframe = "6h"
leverage = 1.0