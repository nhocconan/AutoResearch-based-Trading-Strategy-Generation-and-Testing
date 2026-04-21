#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation on 12h timeframe. Designed for low trade frequency (~12-30/year) to minimize fee drag and work in both bull/bear markets by using 1d HTF for trend/volume context.
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
    
    # === Calculate Camarilla pivot levels from previous 1d bar ===
    # Use previous day's OHLC for today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    camarilla_r1_1d = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_r1_1d_aligned[i]) or np.isnan(camarilla_s1_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        camarilla_r1 = camarilla_r1_1d_aligned[i]
        camarilla_s1 = camarilla_s1_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume spike > 2.0 + price above 1d EMA34
            if price_close > camarilla_r1 and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + volume spike > 2.0 + price below 1d EMA34
            elif price_close < camarilla_s1 and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend/volume confirmation
            if position == 1:
                # Exit long: price breaks below Camarilla S1 OR loss of trend/volume
                if (price_close < camarilla_s1 or 
                    price_close < trend_1d or 
                    vol_spike < 1.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price breaks above Camarilla R1 OR loss of trend/volume
                if (price_close > camarilla_r1 or 
                    price_close > trend_1d or 
                    vol_spike < 1.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0