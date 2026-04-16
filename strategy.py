#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h ATR for volatility filter
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    tr_1h = np.maximum(high_1h - low_1h,
                       np.maximum(np.abs(high_1h - np.roll(close_1h, 1)),
                                  np.abs(low_1h - np.roll(close_1h, 1))))
    tr_1h[0] = high_1h[0] - low_1h[0]
    atr_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # 4h Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h volume spike detection (20-period volume MA)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = vol_4h > (1.5 * vol_ma_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    signals = np.zeros(n)
    warmup = 50
    position = 0
    
    for i in range(warmup, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(volume_spike_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        atr = atr_1h_aligned[i]
        ema_trend = ema_1d_aligned[i]
        vol_spike = volume_spike_4h_aligned[i]
        
        if position == 1:
            if price <= ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if price >= ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        if position == 0:
            if price > upper and vol_spike and atr > 0 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                continue
            elif price < lower and vol_spike and atr > 0 and price < ema_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_VolumeSpike_EMA34Trend"
timeframe = "4h"
leverage = 1.0