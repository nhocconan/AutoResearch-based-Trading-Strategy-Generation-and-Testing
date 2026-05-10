#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Breakout above Camarilla R3 or below S3 with 1d trend filter and volume spike.
# Works in bull/bear by using 1d trend direction and requiring volume confirmation.
# Targets ~25 trades/year to minimize fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d trend: close > EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    
    # 1d volume spike: volume > 1.5 * 20-period EMA
    vol_1d = df_1d['volume'].values
    vol_ema20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_ema20_1d)
    
    # Align 1d trend and volume spike to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We need previous bar's H, L, C
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + (rang * 1.1 / 4)
    camarilla_s3 = prev_close - (rang * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need EMA34 and volume EMA20
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend and volume spike
            if (high[i] > camarilla_r3[i] and
                trend_1d_up_aligned[i] > 0.5 and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend and volume spike
            elif (low[i] < camarilla_s3[i] and
                  trend_1d_up_aligned[i] < 0.5 and  # downtrend
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below R3 or trend reversal
            if (close[i] < camarilla_r3[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above S3 or trend reversal
            if (close[i] > camarilla_s3[i] or
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals