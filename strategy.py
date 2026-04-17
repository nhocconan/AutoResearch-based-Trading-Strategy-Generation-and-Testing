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
    
    # === 1d Donchian channels (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian upper/lower (20-period)
    donch_up = np.full_like(high_1d, np.nan)
    donch_dn = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            donch_up[i] = np.max(high_1d[i-19:i+1])
            donch_dn[i] = np.min(low_1d[i-19:i+1])
        else:
            donch_up[i] = high_1d[i]
            donch_dn[i] = low_1d[i]
    
    # === 1d RSI (14-period) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # === Align 1d indicators to 6h timeframe ===
    donch_up_6h = align_htf_to_ltf(prices, df_1d, donch_up)
    donch_dn_6h = align_htf_to_ltf(prices, df_1d, donch_dn)
    rsi_1d_6h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 6h volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[i]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # 1.5x average volume
    
    signals = np.zeros(n)
    warmup = 30
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_up_6h[i]) or 
            np.isnan(donch_dn_6h[i]) or 
            np.isnan(rsi_1d_6h[i]) or 
            np.isnan(vol_confirm[i])):
            continue
        
        # Long: price breaks above Donchian upper + RSI < 50 (not overbought) + volume confirmation
        if close[i] > donch_up_6h[i] and rsi_1d_6h[i] < 50 and vol_confirm[i]:
            signals[i] = 0.25
        # Short: price breaks below Donchian lower + RSI > 50 (not oversold) + volume confirmation
        elif close[i] < donch_dn_6h[i] and rsi_1d_6h[i] > 50 and vol_confirm[i]:
            signals[i] = -0.25
        # Otherwise flat
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_DonchianBreakout_RSI_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0