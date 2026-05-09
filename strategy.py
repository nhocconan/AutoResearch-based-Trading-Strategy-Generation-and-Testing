#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KeltnerBreakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend and Keltner calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA21 for trend filter
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # 1d ATR for Keltner channels
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
            np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
        )
    )
    atr_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_mid_1d = ema_21_1d  # Using EMA21 as middle line
    keltner_upper_1d = keltner_mid_1d + 2.0 * atr_1d
    keltner_lower_1d = keltner_mid_1d - 2.0 * atr_1d
    
    # Align Keltner channels to 4h
    keltner_mid_4h = align_htf_to_ltf(prices, df_1d, keltner_mid_1d)
    keltner_upper_4h = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_4h = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    
    # Volume filter: above 1.8x 16-period average (16*4h = 2.66 days)
    vol_ma = pd.Series(volume).rolling(window=16, min_periods=16).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 16  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_upper_4h[i]) or np.isnan(keltner_lower_4h[i]) or 
            np.isnan(ema_21_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above Keltner upper with 1d uptrend
            if (close[i] > keltner_upper_4h[i] and 
                close[i] > ema_21_4h[i] and  # 1d uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below Keltner lower with 1d downtrend
            elif (close[i] < keltner_lower_4h[i] and 
                  close[i] < ema_21_4h[i] and  # 1d downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Keltner middle (mean reversion)
            if close[i] < keltner_mid_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Keltner middle (mean reversion)
            if close[i] > keltner_mid_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals