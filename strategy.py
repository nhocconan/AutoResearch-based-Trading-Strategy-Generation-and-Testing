#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Volume Spike + ADX Trend Filter
# - Williams %R (14) on 6h for overbought/oversold signals
# - Long when %R < -80 (oversold) and 1d volume > 1.5x 20-period average (institutional interest)
# - Short when %R > -20 (overbought) and 1d volume > 1.5x 20-period average
# - ADX(14) on 1d > 25 confirms trend strength to avoid chop
# - Volume spike indicates institutional participation; Williams %R captures short-term reversals
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ADX(14) on 1d timeframe
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume average and spike condition
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Williams %R (14) on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Williams %R warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        wr = williams_r[i]
        adx = adx_1d_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # Convert back to boolean
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + volume spike + ADX > 25
            if wr < -80 and vol_spike and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + volume spike + ADX > 25
            elif wr > -20 and vol_spike and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or ADX weakens
            if wr > -50 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or ADX weakens
            if wr < -50 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_VolumeSpike_ADXFilter"
timeframe = "6h"
leverage = 1.0