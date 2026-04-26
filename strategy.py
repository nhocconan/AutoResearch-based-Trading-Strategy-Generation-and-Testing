#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter_v2
Hypothesis: Refine the proven Camarilla R3/S3 breakout by tightening volume confirmation to 3.0x average and adding ATR-based volatility filter (ATR(14) > 0.5*ATR(50)) to avoid low-volume whipsaws. Uses 1d trend filter (close > 1d EMA34) for directional bias. Designed for 25-40 trades/year on 4h by requiring stricter confluence, reducing fee drag while capturing strong trending moves in both bull and bear markets.
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 periods for ATR(50)
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # ATR(14)
    atr_14 = np.zeros(n)
    atr_14[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR(50)
    atr_50 = np.zeros(n)
    atr_50[50] = np.mean(tr[1:51])
    for i in range(51, n):
        atr_50[i] = (atr_50[i-1] * 49 + tr[i]) / 50
    
    # Volatility filter: ATR(14) > 0.5 * ATR(50)
    vol_filter = atr_14 > (0.5 * atr_50)
    
    # Align HTF indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Volume confirmation: volume > 3.0x 20-period average (tighter)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 3.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need all indicators ready
    start_idx = max(34, 50, 20)  # 50 for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_filter_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + 1d uptrend + volume spike + vol filter
            long_signal = (close[i] > camarilla_r3_aligned[i]) and trend_1d_uptrend and volume_spike[i] and vol_filter_aligned[i]
            
            # Short: price breaks below S3 + 1d downtrend + volume spike + vol filter
            short_signal = (close[i] < camarilla_s3_aligned[i]) and trend_1d_downtrend and volume_spike[i] and vol_filter_aligned[i]
            
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
            # Exit: price touches S3 OR 1d trend turns down OR volatility drops
            if (close[i] < camarilla_s3_aligned[i] or not trend_1d_uptrend or not vol_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R3 OR 1d trend turns up OR volatility drops
            if (close[i] > camarilla_r3_aligned[i] or not trend_1d_downtrend or not vol_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter_v2"
timeframe = "4h"
leverage = 1.0