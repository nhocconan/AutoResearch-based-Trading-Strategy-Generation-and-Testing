#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirm
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA trend filter and volume confirmation captures institutional breakouts with lower frequency to minimize fee drag. Works in both bull and bear regimes by requiring alignment with weekly trend. Target: 30-100 trades over 4 years.
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 34:
        return np.zeros(n)
    
    # === Daily Camarilla levels from prior day (HLC of previous 1d bar) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 levels (breakout signals)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Weekly trend filter: 34-period EMA on 1w ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Volume spike filter (20-period on 1d) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        weekly_ema = ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above weekly EMA (bullish trend)
            if price_close > r1 and vol_spike > 2.0 and price_close > weekly_ema:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 + volume spike > 2.0 + price below weekly EMA (bearish trend)
            elif price_close < s1 and vol_spike > 2.0 and price_close < weekly_ema:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit conditions: reverse signal or volume normalization
            if position == 1:
                # Exit long: price breaks below S1 OR volume drops below average
                if price_close < s1 or vol_spike < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price breaks above R1 OR volume drops below average
                if price_close > r1 or vol_spike < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0