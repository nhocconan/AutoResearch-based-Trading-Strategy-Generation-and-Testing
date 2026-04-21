#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation on 12h timeframe.
Designed for low trade frequency (~15-35/year) to minimize fee drag. Uses HTF 1d for trend/volume context.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Typical price for Camarilla calculation (using 1d OHLC) ===
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # === Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4 ===
    # Based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = high_1d - low_1d
    # Avoid division by zero
    range_1d[range_1d == 0] = 1e-10
    
    # Camarilla formulas
    camarilla_r1 = close_1d_prev + range_1d * 1.0/12
    camarilla_s1 = close_1d_prev - range_1d * 1.0/12
    camarilla_r2 = close_1d_prev + range_1d * 2.0/12
    camarilla_s2 = close_1d_prev - range_1d * 2.0/12
    camarilla_r3 = close_1d_prev + range_1d * 3.0/12
    camarilla_s3 = close_1d_prev - range_1d * 3.0/12
    camarilla_r4 = close_1d_prev + range_1d * 4.0/12
    camarilla_s4 = close_1d_prev - range_1d * 4.0/12
    
    # Align all Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike > 2.0 and price above 1d EMA34
            if price_close > r1 and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike > 2.0 and price below 1d EMA34
            elif price_close < s1 and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend/volume
            if position == 1:
                # Exit long if price breaks below S1 or volume drops or trend fails
                if price_close < s1 or vol_spike < 1.2 or price_close < trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price breaks above R1 or volume drops or trend fails
                if price_close > r1 or vol_spike < 1.2 or price_close > trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0