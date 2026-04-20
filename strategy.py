#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Volume Spike + 4h ATR Stop
# - Williams %R(14) on 4h identifies overbought/oversold conditions
# - Long when Williams %R < -80 (oversold) and 1d volume > 1.5x 20-period average
# - Short when Williams %R > -20 (overbought) and 1d volume > 1.5x 20-period average
# - Williams %R captures mean-reversion in ranging markets; volume spike confirms institutional interest
# - ATR-based stop loss limits downside during strong trends
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume analysis
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (1.5 * vol_ma_20)
    vol_spike_4h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate Williams %R(14) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14)
    
    # Calculate ATR(14) for stop loss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(atr[i]) or np.isnan(vol_spike_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        wr = williams_r[i]
        vol_spike_now = vol_spike_4h[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + volume spike
            if wr < -80 and vol_spike_now:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Williams %R overbought (> -20) + volume spike
            elif wr > -20 and vol_spike_now:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 or ATR stop hit
            if wr > -50 or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 or ATR stop hit
            if wr < -50 or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dVolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0