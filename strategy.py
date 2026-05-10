#!/usr/bin/env python3
# 4h_EquiVolume_RSI_Trend
# Hypothesis: On 4h timeframe, combine EquiVolume-weighted RSI with 1d EMA trend filter and volume confirmation.
# EquiVolume weights price by volume to identify institutional interest. RSI(14) > 55 with rising EquiVolume indicates bullish momentum.
# RSI(14) < 45 with falling EquiVolume indicates bearish momentum. 1d EMA50 filter ensures trades align with higher timeframe trend.
# Volume confirmation (current volume > 1.5x 20-period average) reduces false signals.
# Designed for low trade frequency (20-40/year) to minimize fee flood in choppy markets.

name = "4h_EquiVolume_RSI_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EquiVolume calculation: typical price * volume
    typical_price = (high + low + close) / 3.0
    equivolume = typical_price * volume
    
    # RSI(14) on EquiVolume
    rsi_period = 14
    delta = np.diff(equivolume, prepend=equivolume[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    # HTF: 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI > 55, rising EquiVolume (proxy: RSI rising), above 1d EMA50, volume confirmation
            if rsi_val > 55 and rsi[i] > rsi[i-1] and close[i] > ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45, falling EquiVolume (proxy: RSI falling), below 1d EMA50, volume confirmation
            elif rsi_val < 45 and rsi[i] < rsi[i-1] and close[i] < ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI < 50 or breaks below 1d EMA50
            if rsi_val < 50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI > 50 or breaks above 1d EMA50
            if rsi_val > 50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals