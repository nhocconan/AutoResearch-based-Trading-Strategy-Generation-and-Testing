#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 or below S1 with 1-day EMA34 trend filter and volume spike confirmation captures institutional breakouts with low false signals. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop ===
    # 1d for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1-day EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Williams Fractals on 1d for Camarilla levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (H, L, C from prior day)
    # Use shift(1) to ensure we only use completed daily bars
    H = np.roll(high_1d, 1)
    L = np.roll(low_1d, 1)
    C = np.roll(close_1d_arr, 1)
    # First value will be invalid (rolled from last), but alignment handles this
    
    # Camarilla R1 and S1
    camarilla_r1 = C + (H - L) * 1.1 / 12
    camarilla_s1 = C - (H - L) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot points)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume spike on 12h (primary timeframe) ===
    vol_ma_20 = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / np.maximum(vol_ma_20, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_34 = ema_34_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        volume_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above R1 + above daily EMA34 + volume spike
            if price_high > r1 and price_close > ema_34 and volume_spike > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below daily EMA34 + volume spike
            elif price_low < s1 and price_close < ema_34 and volume_spike > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price re-enters Camarilla levels or trend weakens
            if position == 1:
                if price_low < r1 or price_close < ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_high > s1 or price_close > ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3"
timeframe = "12h"
leverage = 1.0