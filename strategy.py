#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Pivot_R1S1_Breakout_Volume_ATRFilter_Tight"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot Points
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = close_1w + range_1w * 1.1 / 12.0
    s1_1w = close_1w - range_1w * 1.1 / 12.0
    
    # Align weekly levels to daily timeframe
    pivot_d = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_d = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_d = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily ATR for volatility and stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(pivot_d[i]) or np.isnan(r1_d[i]) or np.isnan(s1_d[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        pivot = pivot_d[i]
        r1 = r1_d[i]
        s1 = s1_d[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above R1 + volume
            if price > r1 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume
            elif price < s1 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot OR ATR stop (2x ATR from entry)
            if price < pivot or price < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above pivot OR ATR stop (2x ATR from entry)
            if price > pivot or price > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals