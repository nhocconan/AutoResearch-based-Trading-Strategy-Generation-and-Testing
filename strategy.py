#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v3
Hypothesis: Camarilla R1/S1 breakouts on 4h aligned with 1d EMA34 trend and volume spikes capture strong momentum while avoiding chop. Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d Camarilla levels (R1, S1) from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])  # previous day close
    
    camarilla_range = high_1d - low_1d
    r1 = close_1d_prev + 1.1 * camarilla_range / 12
    s1 = close_1d_prev - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h volume: spike > 2.0x 20-period average, avoid low volume < 0.5x 50-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    low_volume = volume < 0.5 * vol_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA34(34), vol MA20(20), vol MA50(50)
    start_idx = max(34, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(vol_ma_50[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend and volume conditions
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        vol_confirmed = volume_spike[i] and not low_volume[i]
        
        # Breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long: breakout above R1 in uptrend with volume confirmation
        if uptrend and vol_confirmed and breakout_r1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short: breakout below S1 in downtrend with volume confirmation
        elif downtrend and vol_confirmed and breakout_s1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit: loss of trend OR entering low volume regime
        elif position == 1 and (not uptrend or low_volume[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or low_volume[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0