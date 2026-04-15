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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period) for trend direction
    highest_20w = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20w = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    highest_20w_6h = align_htf_to_ltf(prices, df_1w, highest_20w)
    lowest_20w_6h = align_htf_to_ltf(prices, df_1w, lowest_20w)
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily Camarilla pivot levels
    daily_range = daily_high - daily_low
    camarilla_h4 = daily_close + daily_range * 1.1 / 2
    camarilla_l4 = daily_close - daily_range * 1.1 / 2
    camarilla_h3 = daily_close + daily_range * 1.1 / 4
    camarilla_l3 = daily_close - daily_range * 1.1 / 4
    
    # Align HTF indicators to 6h timeframe with proper delay
    camarilla_h4_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_6h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_6h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20w_6h[i]) or np.isnan(lowest_20w_6h[i]) or 
            np.isnan(camarilla_h4_6h[i]) or np.isnan(camarilla_l4_6h[i]) or
            np.isnan(camarilla_h3_6h[i]) or np.isnan(camarilla_l3_6h[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly Donchian breakout/breakdown in direction of trend
        # 2. Price near Camarilla H3/L3 levels for mean reversion in ranging markets
        # 3. Volume confirmation: volume > 1.3x 20-period average
        # 4. Volatility filter: ATR > 0.25% of price
        # 5. Discrete position sizing: 0.25
        
        # Calculate 6h volume ratio (current vs 20-period average)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_ratio = volume / (vol_ma_20 + 1e-10)
        
        # Long conditions: Weekly uptrend + price at Camarilla L3 support + volume
        if (close[i] > highest_20w_6h[i] and          # Weekly uptrend (above weekly Donchian high)
            close[i] <= camarilla_l3_6h[i] * 1.005 and  # Near Camarilla L3 support (0.5% tolerance)
            volume_ratio[i] > 1.3 and                 # Volume confirmation
            atr_14[i] > 0.0025 * close[i]):           # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: Weekly downtrend + price at Camarilla H3 resistance + volume
        elif (close[i] < lowest_20w_6h[i] and         # Weekly downtrend (below weekly Donchian low)
              close[i] >= camarilla_h3_6h[i] * 0.995 and  # Near Camarilla H3 resistance (0.5% tolerance)
              volume_ratio[i] > 1.3 and                 # Volume confirmation
              atr_14[i] > 0.0025 * close[i]):           # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyDonchian_CamarillaPivot_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0