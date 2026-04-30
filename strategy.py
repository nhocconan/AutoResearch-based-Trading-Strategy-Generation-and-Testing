#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume confirmation.
# Williams %R measures overbought/oversold conditions: values below -80 = oversold, above -20 = overbought.
# In trending markets (price > 1d EMA34), look for Williams %R to cross above -80 from below with volume for long entries.
# In trending markets (price < 1d EMA34), look for Williams %R to cross below -20 from above with volume for short entries.
# In ranging markets (price near 1d EMA34), fade extreme Williams %R readings (<-90 for long, >-10 for short) with volume confirmation.
# Designed for low trade frequency (~12-37/year) to minimize fee drag. Works in bull/bear via regime adaptation.

name = "6h_WilliamsR_1dEMA34_RegimeAdaptive_VolumeConfirm_v3"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Regime filter: price above/below 1d EMA34 determines trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for Williams %R crossing above -80 from below (momentum long)
                if i > start_idx and williams_r[i-1] <= -80 and curr_williams_r > -80 and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend:
                # In downtrend: look for Williams %R crossing below -20 from above (momentum short)
                if i > start_idx and williams_r[i-1] >= -20 and curr_williams_r < -20 and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging market (near EMA): fade extreme Williams %R readings
                if curr_williams_r < -90 and curr_volume_spike:
                    # Deep oversold: look for long mean reversion
                    signals[i] = 0.25
                    position = 1
                elif curr_williams_r > -10 and curr_volume_spike:
                    # Deep overbought: look for short mean reversion
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R reaches overbought territory (> -20) or loses volume confirmation
            if curr_williams_r >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R reaches oversold territory (< -80) or loses volume confirmation
            if curr_williams_r <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals