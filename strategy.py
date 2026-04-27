#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_HTF
Hypothesis: Camarilla R3/S3 breakout on 12h with 1-week EMA34 trend filter and volume confirmation.
Long when price breaks above R3 with volume spike and 1w trend up; short when price breaks below S3 with volume spike and 1w trend down.
Camarilla levels provide high-probability reversal/breakout points. 1w trend filter ensures trading with the dominant weekly momentum.
Volume spike confirms institutional participation. Designed for 12-30 trades/year on 12h to minimize fee drag while capturing major moves in both bull and bear markets.
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
    
    # Calculate Camarilla levels for 12h: R3, S3, R4, S4
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #            S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using typical formula: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = high - low
    r3 = close + camarilla_range * 1.1 / 4
    s3 = close - camarilla_range * 1.1 / 4
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Camarilla calculation, EMA34, volume average
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with volume spike and 1w trend alignment
            # Long: Price breaks above R3 AND volume spike AND 1w trend up (close > EMA34)
            # Short: Price breaks below S3 AND volume spike AND 1w trend down (close < EMA34)
            long_condition = close_val > r3[i] and vol_spike and close_val > ema_trend
            short_condition = close_val < s3[i] and vol_spike and close_val < ema_trend
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below S3 (reversal) OR 1w trend turns down
            if close_val < s3[i] or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R3 (reversal) OR 1w trend turns up
            if close_val > r3[i] or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_HTF"
timeframe = "12h"
leverage = 1.0