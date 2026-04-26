#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: On 6h timeframe, enter long when price breaks above Camarilla R3 level AND 1d trend is up (close > EMA34) AND volume > 2.0x 20-period average. Enter short when price breaks below S3 level AND 1d trend is down (close < EMA34) AND volume spike. Uses wider Camarilla levels (R3/S3) to capture stronger breakouts with 1d EMA34 trend filter and volume confirmation. Designed for lower trade frequency (12-37/year) with edge in both bull and bear markets via trend alignment and volatility-based entry. Avoids SOL-only bias by requiring BTC/ETH to show similar behavior.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla Pivot Levels (R3, S3) from 1d data
    # Based on previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + ((high-low)*1.1/4), S3 = close - ((high-low)*1.1/4)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (use previous 1d bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (34), volume MA warmup (20), ATR warmup (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # ATR filter: only trade when volatility is normal (not extreme)
        atr_mean = np.mean(atr[max(0, i-50):i]) if i > 0 else atr[i]
        atr_ratio = atr[i] / np.maximum(atr_mean, 1e-10)
        volatility_normal = (atr_ratio > 0.3) & (atr_ratio < 3.0)
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r3 = close[i] > camarilla_r3_aligned[i]
        breakout_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price above R3 + 1d uptrend + volume spike + normal volatility
            long_signal = breakout_above_r3 and trend_uptrend and volume_spike[i] and volatility_normal
            
            # Short: price below S3 + 1d downtrend + volume spike + normal volatility
            short_signal = breakout_below_s3 and trend_downtrend and volume_spike[i] and volatility_normal
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S3 OR trend change to downtrend OR volatility extreme
            if close[i] < camarilla_s3_aligned[i] or not trend_uptrend or not volatility_normal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 OR trend change to uptrend OR volatility extreme
            if close[i] > camarilla_r3_aligned[i] or not trend_downtrend or not volatility_normal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0