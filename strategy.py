# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Hypothesis:
Timeframe: 12h
Primary signal: Breakout above/below 1-day ATR-based volatility bands with volume confirmation
HTF: 1-day ATR for volatility bands and trend filter
Why should work in bull and bear:
- Volatility expansion breakouts capture momentum in both directions
- Volume confirmation filters false breakouts
- ATR-based bands adapt to changing volatility regimes
- Limited to 12h timeframe to reduce trade frequency and fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ATR_Volatility_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ATR on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Upper and lower bands: close ± 1.5 * ATR
    upper_band = close_1d + 1.5 * atr_1d
    lower_band = close_1d - 1.5 * atr_1d
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h timeframe
    upper_band_12h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_12h = align_htf_to_ltf(prices, df_1d, lower_band)
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_12h[i]) or np.isnan(lower_band_12h[i]) or 
            np.isnan(ema50_1d_12h[i]) or np.isnan(vol_avg_1d_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = upper_band_12h[i]
        lower = lower_band_12h[i]
        trend = ema50_1d_12h[i]
        vol_avg = vol_avg_1d_12h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above upper band with volume and above EMA50
            if close[i] > upper and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume and below EMA50
            elif close[i] < lower and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower band or trend reversal
            if close[i] < lower or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper band or trend reversal
            if close[i] > upper or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals