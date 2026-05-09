#!/usr/bin/env python3
# 6h_Chaikin_Relative_Strength_Momentum
# Hypothesis: Chaikin Money Flow (CMF) combined with relative strength vs BTC/ETH shows institutional accumulation/distribution.
# Works in bull/bear: CMF > 0 indicates buying pressure, < 0 selling pressure. Relative strength filters for momentum alignment.
# Uses 12h trend filter and volume confirmation to avoid whipsaws. Target: 50-150 total trades over 4 years.
# Timeframe: 6h, Leverage: 1.0

name = "6h_Chaikin_Relative_Strength_Momentum"
timeframe = "6h"
leverage = 1.0

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
    
    # Chaikin Money Flow (20-period)
    # CMF = sum((Close - Low) - (High - Close)) / (High - Low) * Volume / sum(Volume)
    # Simplified: CMF = sum(((Close - Low) - (High - Close)) * Volume) / sum(Volume)
    # Which equals: sum((2*Close - High - Low) * Volume) / sum(Volume) / (High - Low)
    # But we avoid division by zero
    
    high_low = high - low
    # Avoid division by zero
    valid_hl = high_low != 0
    
    money_flow_multiplier = np.zeros_like(close)
    money_flow_multiplier[valid_hl] = ((2 * close[valid_hl] - high[valid_hl] - low[valid_hl]) / high_low[valid_hl])
    
    money_flow_volume = money_flow_multiplier * volume
    
    # 20-period sums
    mfv_sum = np.full_like(volume, np.nan)
    vol_sum = np.full_like(volume, np.nan)
    
    if len(volume) >= 20:
        mfv_sum[19] = np.sum(money_flow_volume[0:20])
        vol_sum[19] = np.sum(volume[0:20])
        for i in range(20, len(volume)):
            mfv_sum[i] = mfv_sum[i-1] + money_flow_volume[i] - money_flow_volume[i-20]
            vol_sum[i] = vol_sum[i-1] + volume[i] - volume[i-20]
    
    cmf = np.full_like(volume, np.nan)
    valid = (~np.isnan(mfv_sum)) & (~np.isnan(vol_sum)) & (vol_sum != 0)
    cmf[valid] = mfv_sum[valid] / vol_sum[valid]
    
    # Get 12h trend filter (EMA 50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (ema_50_12h[i-1] * 49 + close_12h[i]) / 50
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cmf[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: CMF > 0.1 (buying pressure) AND uptrend (price > EMA50) AND volume spike
            if (cmf[i] > 0.1 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: CMF < -0.1 (selling pressure) AND downtrend (price < EMA50) AND volume spike
            elif (cmf[i] < -0.1 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CMF turns negative OR trend reversal (price < EMA50)
            if cmf[i] < -0.05 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CMF turns positive OR trend reversal (price > EMA50)
            if cmf[i] > 0.05 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals