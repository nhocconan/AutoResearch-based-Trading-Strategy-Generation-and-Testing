#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Control_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # === Weekly: Camarilla R1 and S1 levels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Calculate Camarilla levels
    r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12.0
    s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12.0
    
    # Align weekly levels to daily
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === Daily: Price, volume, ATR ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ratio_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr[max(0, i-49):i+1]) if i >= 1 else np.nan
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and volatility confirmation
            if (close_val > r1_val and
                vol_ratio_val > 1.5 and    # Volume confirmation
                vol_filter):               # Volatility filter
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and volatility confirmation
            elif (close_val < s1_val and
                  vol_ratio_val > 1.5 and    # Volume confirmation
                  vol_filter):               # Volatility filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls back below weekly R1 or volatility drops
            if (close_val < r1_val) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above weekly S1 or volatility drops
            if (close_val > s1_val) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals