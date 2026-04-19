#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 12h EMA34 breakout with volume confirmation and 4h ATR filter.
# Uses 12h EMA34 as trend filter to avoid counter-trend trades.
# Breakouts occur when price crosses EMA34 with volume > 1.5x 20-period average.
# 4h ATR filter ensures volatility is sufficient (ATR > 1.5x 50-period MA) to avoid chop.
# Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years.

name = "1h_12hEMA34_VolumeBreakout_4hATRFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 4h data for ATR filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR(14)
    tr1 = np.maximum(high_4h[1:], close_4h[:-1]) - np.minimum(low_4h[1:], close_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_4h = pd.Series(atr_4h).rolling(window=50, min_periods=50).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    atr_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(atr_ma_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_12h_aligned[i]
        atr = atr_4h_aligned[i]
        atr_ma = atr_ma_4h_aligned[i]
        hour = hours[i]
        
        # Session filter: only trade 08-20 UTC
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 1.5x MA to avoid chop
        vol_filter = atr > 1.5 * atr_ma
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price above EMA34 with volume and vol filter
            if price > ema and volume_confirmed and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Price below EMA34 with volume and vol filter
            elif price < ema and volume_confirmed and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA34
            if price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price closes above EMA34
            if price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals