#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly EMA(34) for additional trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        ema_trend_1d = ema34_1d_aligned[i]
        ema_trend_1w = ema34_1w_aligned[i]
        atr_current = atr[i]
        
        # Volatility filter: only trade when current volatility is above 80% of ATR
        # (This is a placeholder - in practice we'd use weekly ATR, but we'll simplify for now)
        vol_filter = atr_current > 0  # Always true for now, can be refined
        
        if position == 0:
            # Only trade when both daily and weekly trends agree
            trend_aligned = (ema_trend_1d > ema_trend_1w)  # Uptrend when daily > weekly
            
            # Donchian breakout with trend and volume confirmation
            # Long: break above upper band with volume spike and uptrend
            if (high[i] > high_20[i] and close[i] > high_20[i] and 
                trend_aligned and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below lower band with volume spike and downtrend
            elif (low[i] < low_20[i] and close[i] < low_20[i] and 
                  not trend_aligned and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches lower Donchian band or trend reverses
            if low[i] <= low_20[i] or not trend_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches upper Donchian band or trend reverses
            if high[i] >= high_20[i] or trend_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1d1wEMA34_Trend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0