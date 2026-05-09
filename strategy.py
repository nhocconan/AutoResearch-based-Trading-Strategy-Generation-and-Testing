#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Enhanced"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation (R1, S1)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1)
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12  # R1 = C + 1.1*(H-L)/12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12  # S1 = C - 1.1*(H-L)/12
    
    # Trend filter: 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current 12h volume > 1.5 * 20-period average (stricter)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Choppiness regime filter: avoid choppy markets (trend-following only)
    # Use 1d ATR and ADX-like measure
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ADX-like trend strength (simplified)
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift()) > (df_1d['low'].shift() - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(), 0), 0)
    minus_dm = np.where((df_1d['low'].shift() - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift()), 
                        np.maximum(df_1d['low'].shift() - df_1d['low'], 0), 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr14
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Trend filter: ADX > 25 for trending market
    trend_filter = adx > 25
    
    # Align all to 12h (primary timeframe)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_filter_12h = align_htf_to_ltf(prices, df_1d, trend_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20, 14)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(ema34_1d_12h[i]) or np.isnan(volume_filter[i]) or
            np.isnan(trend_filter_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_12h[i]
        s1_val = s1_12h[i]
        trend = ema34_1d_12h[i]
        vol_filter = volume_filter[i]
        trending = trend_filter_12h[i]
        
        if position == 0:
            # Enter long: break above R1 with volume, trend, and in trending market
            if close[i] > r1_val and close[i] > trend and vol_filter and trending:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume, trend, and in trending market
            elif close[i] < s1_val and close[i] < trend and vol_filter and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 or loss of trend
            if close[i] < s1_val or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 or loss of trend
            if close[i] > r1_val or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals