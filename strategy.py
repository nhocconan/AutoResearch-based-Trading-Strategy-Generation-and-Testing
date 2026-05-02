#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX(14) regime filter and volume confirmation
# Uses 12h primary timeframe for Camarilla pivot breakout signals
# 1d ADX > 25 confirms trending market (avoids ranging/choppy conditions)
# Volume confirmation (2.0x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by only trading in trending regimes (ADX > 25)

name = "12h_Camarilla_R3S3_Breakout_1dADX25_Trend_Volume_v1"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high_1d.diff()
    dm_minus = low_1d.diff() * -1  # Invert to get positive values
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    tr_ma = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_ma = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_ma = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_ma / tr_ma)
    di_minus = 100 * (dm_minus_ma / tr_ma)
    
    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_1d = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get daily OHLC from 1d data for Camarilla levels
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_R3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_S3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (1d -> 12h: 2 bars per day)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # ADX regime filter: only trade when ADX > 25 (trending market)
        trending_market = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Camarilla breakout long: price > R3
            # Camarilla breakout short: price < S3
            breakout_long = close[i] > camarilla_R3_aligned[i]
            breakout_short = close[i] < camarilla_S3_aligned[i]
            
            if breakout_long and trending_market and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif breakout_short and trending_market and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Camarilla breakdown (price < S3) or loss of trend
            if close[i] < camarilla_S3_aligned[i] or adx_1d_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Camarilla breakout (price > R3) or loss of trend
            if close[i] > camarilla_R3_aligned[i] or adx_1d_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals