#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ADXFilter_v3
Hypothesis: 4h breakouts of 1d Camarilla R3/S3 levels with 1d EMA34 trend filter, volume confirmation (>2x 20-bar MA), and 1d ADX > 25 regime. Uses discrete sizing (0.25) to minimize fee churn. Designed for lower trade frequency (~20-40 trades/year) to avoid overtrading failures seen in similar strategies.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d bar's OHLC for Camarilla levels (R3/S3)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1d ADX regime filter (ADX > 25 for trending market)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d_arr[i-1]), abs(low_1d[i] - close_1d_arr[i-1]))
    
    period = 14
    alpha = 1.0 / period
    atr_1d = np.zeros(len(df_1d))
    plus_dm_smooth = np.zeros(len(df_1d))
    minus_dm_smooth = np.zeros(len(df_1d))
    
    if len(df_1d) >= period + 1:
        atr_1d[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, len(df_1d)):
        atr_1d[i] = atr_1d[i-1] * (1 - alpha) + alpha * tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - alpha) + alpha * plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - alpha) + alpha * minus_dm[i]
    
    plus_di_1d = np.zeros(len(df_1d))
    minus_di_1d = np.zeros(len(df_1d))
    dx_1d = np.zeros(len(df_1d))
    
    for i in range(period, len(df_1d)):
        if atr_1d[i] != 0:
            plus_di_1d[i] = 100 * plus_dm_smooth[i] / atr_1d[i]
            minus_di_1d[i] = 100 * minus_dm_smooth[i] / atr_1d[i]
        dx_1d[i] = 100 * abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i]) if (plus_di_1d[i] + minus_di_1d[i]) != 0 else 0
    
    adx_1d = np.zeros(len(df_1d))
    if len(df_1d) >= period*2 + 1:
        adx_1d[period*2] = np.mean(dx_1d[period+1:period*2+1]) if len(dx_1d[period+1:period*2+1]) > 0 else 0
    
    for i in range(period*2+1, len(df_1d)):
        adx_1d[i] = adx_1d[i-1] * (1 - alpha) + alpha * dx_1d[i]
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of calculations
    start_idx = max(20, 34, period*2+1)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        trending_regime = adx_val > 25
        
        long_entry = (close_val > r3_val) and bullish_1d and vol_spike and trending_regime
        short_entry = (close_val < s3_val) and bearish_1d and vol_spike and trending_regime
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r3_val or not bullish_1d or not trending_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s3_val or not bearish_1d or not trending_regime):
            signals[i] = 0.0
            position = 0
        else:
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ADXFilter_v3"
timeframe = "4h"
leverage = 1.0