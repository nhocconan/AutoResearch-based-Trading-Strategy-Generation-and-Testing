#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and volume spike confirmation.
Uses 1h primary timeframe with 4h HTF for trend direction and volume context.
Designed for low trade frequency (~15-35/year) to minimize fee drag while capturing intraday breakouts
with institutional levels. Works in both bull and bear markets by following 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # === 4h trend filter: 50-period EMA ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 4h volume average (20-period) for spike detection ===
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h[np.isnan(vol_ma_4h)] = 1.0  # avoid division by zero
    vol_ratio_4h = volume_4h / vol_ma_4h
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # === Previous day's Camarilla levels (using 1d data for pivot calculation) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = high_1d - low_1d
    camarilla_R1 = close_1d + camarilla_range * 1.1 / 12
    camarilla_S1 = close_1d - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ratio_4h_aligned[i]) or
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        trend_4h = ema_50_4h_aligned[i]
        vol_spike = vol_ratio_4h_aligned[i]
        camarilla_R1 = camarilla_R1_aligned[i]
        camarilla_S1 = camarilla_S1_aligned[i]
        
        if position == 0:
            # Long: price breaks above camarilla R1 + volume spike > 2.0 + price above 4h EMA50
            if price_close > camarilla_R1 and vol_spike > 2.0 and price_close > trend_4h:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below camarilla S1 + volume spike > 2.0 + price below 4h EMA50
            elif price_close < camarilla_S1 and vol_spike > 2.0 and price_close < trend_4h:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend/volume
            if position == 1:
                # Exit long: price breaks below camarilla S1 OR loss of 4h uptrend
                if price_close < camarilla_S1 or price_close < trend_4h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: price breaks above camarilla R1 OR loss of 4h downtrend
                if price_close > camarilla_R1 or price_close > trend_4h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0