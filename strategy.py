#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d Volume Spike + Chop Filter
# - Williams %R(14) on 12h for overbought/oversold signals
# - Long when %R < -80 (oversold) and 1d volume > 1.5x 20-period average (volume spike)
# - Short when %R > -20 (overbought) and 1d volume > 1.5x 20-period average (volume spike)
# - Only trade when 1d Choppiness Index > 61.8 (range market) to avoid trending whipsaws
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and chop calculations
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d volume spike indicator (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_ma_20 * 1.5)
    
    # Calculate 1d Choppiness Index (14-period)
    atr_14 = []
    for i in range(len(high_1d)):
        if i < 14:
            atr_14.append(np.nan)
        else:
            tr = np.max([
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            ])
            atr_14.append(tr)
    atr_14 = np.array(atr_14)
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_mask = chop > 61.8  # Range market condition
    
    # Align 1d indicators to 12h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Williams %R (14) on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Williams %R warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        wr = williams_r[i]
        vol_spike = vol_spike_aligned[i] > 0.5  # Convert back to boolean
        in_chop = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + volume spike + chop market
            if wr < -80 and vol_spike and in_chop:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + volume spike + chop market
            elif wr > -20 and vol_spike and in_chop:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or volatility spike ends
            if wr > -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or volatility spike ends
            if wr < -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dVolSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0