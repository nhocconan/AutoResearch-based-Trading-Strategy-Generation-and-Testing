#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and ADX regime filter
# In trending markets (1d ADX > 25): breakout above R3 or below S3 with volume confirmation
# In ranging markets (1d ADX <= 25): fade extreme levels (R4/S4) with volume confirmation
# Uses discrete sizing (0.25) to minimize fee churn. Designed for 4h timeframe targeting 75-200 trades over 4 years.
# BTC/ETH edge: Camarilla pivots from 1d capture institutional levels; ADX regime avoids whipsaws; volume confirms participation.

name = "4h_Camarilla_1dADX_Volume_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Shift by 1 to avoid look-ahead: use previous day's close to calculate today's levels
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # True range for Camarilla calculation
    tr = pd.Series([
        max(h - l, abs(h - pc), abs(l - pc))
        for h, l, pc in zip(prev_high, prev_low, prev_close)
    ])
    atr_1d = tr.rolling(window=5, min_periods=5).mean()
    
    # Camarilla levels (based on previous day's range)
    range_1d = prev_high - prev_low
    camarilla_r4 = prev_close + range_1d * 1.500
    camarilla_r3 = prev_close + range_1d * 1.250
    camarilla_s3 = prev_close - range_1d * 1.250
    camarilla_s4 = prev_close - range_1d * 1.500
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Calculate 1d ADX (14-period)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<=25) or trending (ADX>25)
            if adx_aligned[i] <= 25:
                # Ranging market: fade extreme levels (R4/S4)
                if close[i] <= s4_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= r4_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: breakout of R3/S3 with volume confirmation
                if close[i] > r3_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s3_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint OR ADX weakens (<20) OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] >= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR ADX weakens (<20) OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] <= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals