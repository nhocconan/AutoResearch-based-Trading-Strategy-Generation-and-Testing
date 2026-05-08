# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume spike.
# Long when Williams %R crosses above -80 from oversold AND 1d EMA200 uptrend AND volume spike.
# Short when Williams %R crosses below -20 from overbought AND 1d EMA200 downtrend AND volume spike.
# Williams %R(14) measures momentum extremes; 1d EMA200 filters trend direction; volume spike confirms momentum.
# Designed to work in both bull and bear markets by using 1d trend for direction and Williams %R for mean reversion within trend.

name = "6h_WilliamsR_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and volume context
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Williams %R(14) on 6h data
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Daily EMA200 for trend direction
    close_d = df_d['close'].values
    ema200_d = pd.Series(close_d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_prev_d = np.roll(ema200_d, 1)
    ema200_prev_d[0] = ema200_d[0]
    ema200_uptrend = ema200_d > ema200_prev_d
    ema200_downtrend = ema200_d < ema200_prev_d
    
    # Daily volume spike: current volume > 2.0 x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_spike_d = volume_d > (2.0 * vol_ma20_d)
    
    # Align daily indicators to 6h timeframe
    ema200_uptrend_aligned = align_htf_to_ltf(prices, df_d, ema200_uptrend)
    ema200_downtrend_aligned = align_htf_to_ltf(prices, df_d, ema200_downtrend)
    volume_spike = align_htf_to_ltf(prices, df_d, volume_spike_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(williams_period, 200, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or 
            np.isnan(ema200_uptrend_aligned[i]) or np.isnan(ema200_downtrend_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (exit oversold) AND 1d uptrend AND volume spike
            williams_cross_up = (williams_r[i-1] <= -80) and (williams_r[i] > -80)
            long_cond = williams_cross_up and ema200_uptrend_aligned[i] and volume_spike[i]
            
            # Short conditions: Williams %R crosses below -20 (exit overbought) AND 1d downtrend AND volume spike
            williams_cross_down = (williams_r[i-1] >= -20) and (williams_r[i] < -20)
            short_cond = williams_cross_down and ema200_downtrend_aligned[i] and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum weakening) OR 1d trend turns down
            williams_cross_down_mid = (williams_r[i-1] >= -50) and (williams_r[i] < -50)
            trend_turned_down = ema200_uptrend_aligned[i-1] and not ema200_uptrend_aligned[i]
            if williams_cross_down_mid or trend_turned_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum weakening) OR 1d trend turns up
            williams_cross_up_mid = (williams_r[i-1] <= -50) and (williams_r[i] > -50)
            trend_turned_up = ema200_downtrend_aligned[i-1] and not ema200_downtrend_aligned[i]
            if williams_cross_up_mid or trend_turned_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals