#!/usr/bin/env python3
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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on prior day)
    # Camarilla uses prior day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day values (shifted by 1)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Resistance levels
    r1 = pivot + (range_ * 1.1 / 12)
    r2 = pivot + (range_ * 1.1 / 6)
    r3 = pivot + (range_ * 1.1 / 4)
    r4 = pivot + (range_ * 1.1 / 2)
    
    # Support levels
    s1 = pivot - (range_ * 1.1 / 12)
    s2 = pivot - (range_ * 1.1 / 6)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h Donchian channels (20-period) for breakout confirmation
    df_6h = get_htf_data(prices, '6h')  # Get actual 6h data for Donchian calculation
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    upper_20_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_20_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align 6h Donchian to 6h (same timeframe, no alignment needed but keep for consistency)
    upper_20_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_20_6h)
    lower_20_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_20_6h)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: avoid low-volume periods (22-02 UTC typically quieter)
    hours = prices.index.hour
    in_session = (hours >= 2) & (hours <= 22)  # Trade 20:00-02:00 UTC avoided
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(upper_20_6h_aligned[i]) or 
            np.isnan(lower_20_6h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above 6h Donchian upper (20) - bullish breakout
        # 2. Price above Camarilla R3 (strong resistance turned support)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_6h_aligned[i] and
            close[i] > r3_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 6h Donchian lower (20) - bearish breakdown
        # 2. Price below Camarilla S3 (strong support turned resistance)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_6h_aligned[i] and
              close[i] < s3_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_CamarillaR3S3_Donchian20_Breakout_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0