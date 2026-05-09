#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Filtered"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla R3 and S3 from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + camarilla_range * 1.250
    s3 = close_1d - camarilla_range * 1.250
    
    # Align Camarilla R3 and S3 to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike filter: current volume > 2.0 * 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness regime filter: avoid choppy markets
    # Calculate Choppiness Index on 4h data
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((highest_high - lowest_low) / (atr_safe * atr_period)) / np.log10(atr_period)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # Neutral when no range
    
    # Trending market: CHOP < 38.2, Choppy market: CHOP > 61.8
    trending_market = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, atr_period)  # Need enough data for EMA34 and ATR
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(trending_market[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        is_trending = trending_market[i]
        
        if position == 0:
            # Enter long: Close > R3 and price above 1d EMA34 with volume spike in trending market
            if close[i] > r3_aligned[i] and close[i] > ema_1d and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S3 and price below 1d EMA34 with volume spike in trending market
            elif close[i] < s3_aligned[i] and close[i] < ema_1d and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S3 or trend breaks (price < 1d EMA34) or market becomes choppy
            if close[i] < s3_aligned[i] or close[i] < ema_1d or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R3 or trend breaks (price > 1d EMA34) or market becomes choppy
            if close[i] > r3_aligned[i] or close[i] > ema_1d or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals