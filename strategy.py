#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d Donchian channels for structure
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    dh_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # 1d weekly pivot points (using previous week)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1/S1
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    wp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    wr1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    ws1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 60-period EMA for trend filter (on 6h close)
    close = prices['close'].values
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume confirmation
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or
            np.isnan(wp_aligned[i]) or np.isnan(wr1_aligned[i]) or np.isnan(ws1_aligned[i]) or
            np.isnan(ema60[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = dh_20_aligned[i]
        dl = dl_20_aligned[i]
        wp = wp_aligned[i]
        wr1 = wr1_aligned[i]
        ws1 = ws1_aligned[i]
        ema = ema60[i]
        vol = vol_surge[i]
        close_i = close[i]
        
        if position == 0:
            # Long: Break above Donchian high + above weekly pivot + above EMA60 + volume surge
            if (close_i > dh and close_i > wp and close_i > ema and vol):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + below weekly pivot + below EMA60 + volume surge
            elif (close_i < dl and close_i < wp and close_i < ema and vol):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to weekly pivot or trend fails
            if position == 1:
                if close_i < wp or close_i < ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_i > wp or close_i > ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0