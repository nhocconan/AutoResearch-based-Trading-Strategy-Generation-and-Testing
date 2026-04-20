#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Pivot_R1S1_Breakout_Volume_Control"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h: Calculate daily pivots from previous day ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous day's pivot calculation (shifted by 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3.0
    # R1 and S1
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume ratio (current vs 24-period average)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma24 > 0, vol_ma24, np.nan)
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ratio_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr[max(0, i-49):i+1]) if i >= 1 else np.nan
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and volatility filter
            if (close_val > r1_val and   # Break above R1
                vol_ratio_val > 1.5 and    # Volume confirmation
                vol_filter):               # Volatility filter
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and volatility filter
            elif (close_val < s1_val and   # Break below S1
                  vol_ratio_val > 1.5 and    # Volume confirmation
                  vol_filter):               # Volatility filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below pivot point or volatility drops
            if (close_val < pp_val) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above pivot point or volatility drops
            if (close_val > pp_val) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals