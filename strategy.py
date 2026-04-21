#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 or below S1 from the prior day, with 1-day EMA34 trend filter and volume spike confirmation, captures momentum breakouts with low trade frequency. Designed for 4h TF to target 75-200 total trades over 4 years (19-50/year) to minimize fee drag and improve generalization to bear markets (2025+). Uses volume > 1.5x 20-period MA as confirmation filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter and Camarilla levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1-day EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Previous day's OHLC for Camarilla levels ===
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (inner levels for more frequent but still filtered signals)
    camarilla_r1 = prev_close + prev_range * 1.1 / 12
    camarilla_s1 = prev_close - prev_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume confirmation: 20-period MA of volume ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_34 = ema_34_1d_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        volume_val = volume[i]
        
        if position == 0:
            # Long: price breaks above R1 + above daily EMA34 + volume spike
            if price_high > r1_level and price_close > ema_34 and volume_val > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 + below daily EMA34 + volume spike
            elif price_low < s1_level and price_close < ema_34 and volume_val > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit: trend weakens (price crosses back below/above EMA34)
            if position == 1:
                if price_close < ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0