#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud + 1d Kijun Filter + Volume Spike
# Hypothesis: Ichimoku signals aligned with daily Kijun (base line) direction
# and volume spikes capture trend momentum while avoiding false signals in chop.
# Works in bull/bear via cloud direction filter. Target: 15-35 trades/year (60-140 total).

name = "6h_ichimoku_1d_kijun_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Ichimoku components (9, 26, 52)
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # 6h Ichimoku components (9, 26, 52)
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    senkou_a_6h = ((tenkan_6h + kijun_6h) / 2)
    senkou_b_6h = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low).rolling(window=52, min_periods=52).min()) / 2)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Cloud top and bottom (6h)
        cloud_top_6h = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom_6h = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR trend turns bearish (price < daily kijun)
            if close[i] < cloud_bottom_6h[i] or close[i] < kijun_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR trend turns bullish (price > daily kijun)
            if close[i] > cloud_top_6h[i] or close[i] > kijun_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish: price above cloud, tenkan > kijun, and above daily kijun
                if (close[i] > cloud_top_6h[i] and 
                    tenkan_6h[i] > kijun_6h[i] and 
                    close[i] > kijun_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Bearish: price below cloud, tenkan < kijun, and below daily kijun
                elif (close[i] < cloud_bottom_6h[i] and 
                      tenkan_6h[i] < kijun_6h[i] and 
                      close[i] < kijun_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals