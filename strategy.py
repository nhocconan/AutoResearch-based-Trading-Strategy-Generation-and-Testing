#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX regime filter and volume confirmation
# Camarilla levels provide precise intraday support/resistance. ADX > 25 ensures trending market.
# Volume spike (>1.8x 20 EMA) confirms institutional participation. Discrete sizing 0.25 limits risk.
# Works in bull/bear: ADX filter avoids ranging markets. Target: 80-180 trades over 4 years.

name = "12h_Camarilla_R3S3_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up = high_1d.diff()
    down = low_1d.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    
    # Smoothed DM
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d)
    
    # DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align 1d ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.8 x 20-period EMA
        volume_confirm = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0 and trending:
            # Long conditions: price breaks above Camarilla R3 + volume spike
            if close[i] > r3_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + volume spike
            elif close[i] < s3_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot (R4/S4 midpoint) OR ADX weakens
            camarilla_pivot = (r3_aligned[i] + s3_aligned[i]) / 2.0  # Approximate pivot
            if (close[i] < camarilla_pivot or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot OR ADX weakens
            camarilla_pivot = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] > camarilla_pivot or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals