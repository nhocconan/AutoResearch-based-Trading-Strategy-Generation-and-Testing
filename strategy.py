#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with volume spike and 1d ADX trend filter.
Long when price breaks above Camarilla R1 AND volume > 2.0x average AND 1d ADX > 25 (trending up).
Short when price breaks below Camarilla S1 AND volume > 2.0x average AND 1d ADX > 25 (trending down).
Exit when price reverts to Camarilla H3/L3 level (mean reversion within the day).
Uses 12h for price/volume, 1d for ADX to avoid whipsaw in ranging markets.
Targets 50-150 total trades over 4 years (12-37/year). Camarilla levels provide precise intraday support/resistance,
volume confirmation reduces fakeouts, ADX ensures we only trade in trending markets.
Works in bull markets (captures uptrends with ADX>25) and bear markets (captures downtrends with ADX>25).
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
    
    # Get 12h data for Camarilla levels and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels on 12h timeframe (based on prior 12h bar)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    close_series = pd.Series(close_12h)
    
    # Use prior bar's high/low/close to avoid look-ahead
    prior_high = high_series.shift(1)
    prior_low = low_series.shift(1)
    prior_close = close_series.shift(1)
    
    # Calculate Camarilla levels
    camarilla_r1 = prior_close + (prior_high - prior_low) * 1.1 / 12
    camarilla_s1 = prior_close - (prior_high - prior_low) * 1.1 / 12
    camarilla_h3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    camarilla_l3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Calculate volume average (20-period) on 12h
    volume_series = pd.Series(volume_12h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d timeframe
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_1d_series - high_1d_series.shift(1)
    down_move = low_1d_series.shift(1) - low_1d_series
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 12h Camarilla levels, volume MA, and 1d ADX to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1.values)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3.values)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 2.0x avg AND ADX > 25 (strong uptrend)
            if price > r1 and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 2.0x avg AND ADX > 25 (strong downtrend)
            elif price < s1 and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla H3 (mean reversion toward midpoint)
            if price < h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla L3 (mean reversion toward midpoint)
            if price > l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0