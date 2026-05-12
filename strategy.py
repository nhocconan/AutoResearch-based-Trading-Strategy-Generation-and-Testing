#!/usr/bin/env python3
name = "12h_Vortex_Trend_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # === 1d Data for trend and volume ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d EMA34 trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 1d Volume MA for spike detection ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === Vortex Indicator (14-period) on 12h ===
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    vm = np.abs(high - np.roll(low, 1))
    vp = np.abs(low - np.roll(high, 1))
    vm[0] = high[0] - low[0]
    vp[0] = high[0] - low[0]
    
    # Sum over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm14 = pd.Series(vm).rolling(window=14, min_periods=14).sum().values
    vp14 = pd.Series(vp).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm14 / tr14
    vi_minus = vp14 / tr14
    
    # === Align 1d data to 12h ===
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(vi_plus[i]) or
            np.isnan(vi_minus[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- + above 1d EMA34 + volume spike
            if (vi_plus[i] > vi_minus[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > (vol_ma_1d_aligned[i] * 2.0)):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ + below 1d EMA34 + volume spike
            elif (vi_minus[i] > vi_plus[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > (vol_ma_1d_aligned[i] * 2.0)):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VI- > VI+ (trend reversal) or below 1d EMA34
            if vi_minus[i] > vi_plus[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VI+ > VI- (trend reversal) or above 1d EMA34
            if vi_plus[i] > vi_minus[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals