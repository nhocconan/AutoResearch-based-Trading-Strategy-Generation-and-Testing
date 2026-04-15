#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

"""
Hypothesis: 6h trend following with 1d Supertrend filter and 1w volume confirmation.
- 1d Supertrend (ATR=10, multiplier=3) determines trend direction (avoid counter-trend trades)
- 6h price breaks above/below 20-period Donchian channel with volume > 1.5x 20-bar median
- Weekly volume filter: require current volume > 1.2x 4-week average volume to ensure institutional participation
- Designed for low trade frequency (~20-40/year) to minimize fee impact in ranging/bear markets
- Works in bull (trend continuation) and bear (avoid false breaks via Supertrend filter)
"""

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Supertrend for trend filter
    df_1d = get_htf_data(prices, '1d')
    hl2_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    upper_basic_1d = hl2_1d + 3 * atr_1d
    lower_basic_1d = hl2_1d - 3 * atr_1d
    
    upper_final_1d = np.full_like(hl2_1d, np.nan)
    lower_final_1d = np.full_like(hl2_1d, np.nan)
    for i in range(1, len(hl2_1d)):
        upper_final_1d[i] = upper_basic_1d[i] if (upper_basic_1d[i] < upper_final_1d[i-1] or df_1d['close'].values[i-1] > upper_final_1d[i-1]) else upper_final_1d[i-1]
        lower_final_1d[i] = lower_basic_1d[i] if (lower_basic_1d[i] > lower_final_1d[i-1] or df_1d['close'].values[i-1] < lower_final_1d[i-1]) else lower_final_1d[i-1]
    
    supertrend_1d = np.full_like(hl2_1d, np.nan)
    for i in range(1, len(hl2_1d)):
        if i == 1:
            supertrend_1d[i] = upper_final_1d[i]
        else:
            if supertrend_1d[i-1] == upper_final_1d[i-1] and df_1d['close'].values[i] <= upper_final_1d[i]:
                supertrend_1d[i] = upper_final_1d[i]
            elif supertrend_1d[i-1] == upper_final_1d[i-1] and df_1d['close'].values[i] > upper_final_1d[i]:
                supertrend_1d[i] = lower_final_1d[i]
            elif supertrend_1d[i-1] == lower_final_1d[i-1] and df_1d['close'].values[i] >= lower_final_1d[i]:
                supertrend_1d[i] = lower_final_1d[i]
            elif supertrend_1d[i-1] == lower_final_1d[i-1] and df_1d['close'].values[i] < lower_final_1d[i]:
                supertrend_1d[i] = upper_final_1d[i]
            else:
                supertrend_1d[i] = supertrend_1d[i-1]
    
    supertrend_dir_1d = np.where(df_1d['close'].values > supertrend_1d, 1, -1)
    supertrend_dir_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_dir_1d)
    
    # 6h Donchian channel (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume filters
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold_6h = 1.5 * vol_median_20
    
    # 1w volume filter
    df_1w = get_htf_data(prices, '1w')
    vol_avg_4w = pd.Series(df_1w['volume'].values).rolling(window=4, min_periods=4).mean()
    vol_avg_4w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_4w)
    vol_threshold_1w = 1.2 * vol_avg_4w_aligned
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_threshold_6h[i]) or np.isnan(supertrend_dir_1d_aligned[i]) or
            np.isnan(vol_threshold_1w[i])):
            continue
        
        # Long: price breaks above Donchian upper + volume + uptrend (1d Supertrend) + weekly volume
        if (close[i] > highest_high[i] and volume[i] > vol_threshold_6h[i] and 
            supertrend_dir_1d_aligned[i] == 1 and volume[i] > vol_threshold_1w[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian lower + volume + downtrend (1d Supertrend) + weekly volume
        elif (close[i] < lowest_low[i] and volume[i] > vol_threshold_6h[i] and 
              supertrend_dir_1d_aligned[i] == -1 and volume[i] > vol_threshold_1w[i]):
            signals[i] = -0.25
        
        # Exit: price returns to mid-channel (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (highest_high[i] + lowest_low[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (highest_high[i] + lowest_low[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6d_Supertrend_Donchian_Volume"
timeframe = "6h"
leverage = 1.0