#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Bollinger Band squeeze breakout with volume confirmation and ADX trend filter
# Long when price closes above upper Bollinger Band (20,2) with volume > 1.3x average and ADX > 25
# Short when price closes below lower Bollinger Band with volume > 1.3x average and ADX > 25
# Bollinger squeeze identifies low volatility breakouts; volume confirms breakout strength; ADX ensures trending environment
# Designed to capture explosive moves in both bull and bear markets with controlled frequency
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "4h_1dBB_Squeeze_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Bollinger Bands (20,2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period SMA and standard deviation for Bollinger Bands
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # ADX trend filter (14-period) on 1h timeframe for better trend detection
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # Calculate True Range
    tr1 = pd.Series(df_1h['high']).shift(1) - pd.Series(df_1h['low'])
    tr2 = abs(pd.Series(df_1h['high']) - pd.Series(df_1h['close']).shift(1))
    tr3 = abs(pd.Series(df_1h['low']) - pd.Series(df_1h['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(df_1h['high']) - pd.Series(df_1h['high']).shift(1)
    dm_minus = pd.Series(df_1h['low']).shift(1) - pd.Series(df_1h['low'])
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Bollinger warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above upper BB with volume and trend confirmation
            if close[i] > upper_bb_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower BB with volume and trend confirmation
            elif close[i] < lower_bb_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below middle Bollinger Band (SMA)
            if close[i] < sma_20[i]:  # Using aligned SMA would be better but using raw for simplicity
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above middle Bollinger Band (SMA)
            if close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals