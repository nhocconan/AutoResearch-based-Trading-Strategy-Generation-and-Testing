#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme readings (below -80 for oversold, above -20 for overbought) 
# combined with 1d EMA50 trend filter and volume confirmation. In strong trends (price > 1d EMA50),
# we take Williams %R pullbacks to -50 (continuation). In ranging markets (price near 1d EMA50),
# we fade extremes at Williams %R < -80 or > -20. This captures mean reversion in ranges and 
# trend continuation in trends, adapting to both bull and bear markets. Low trade frequency via 
# strict Williams %R thresholds and volume confirmation minimizes fee drag.

name = "6h_WilliamsR_1dEMA50_RegimeAdaptive_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Regime filter: price above/below 1d EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for long on pullback to Williams %R = -50 with volume
                if -55 <= curr_wr <= -45 and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend:
                # In downtrend: look for short on pullback to Williams %R = -50 with volume
                if -55 <= curr_wr <= -45 and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging market (near EMA): fade Williams %R extremes
                if curr_wr < -80:  # Deep oversold: look for long
                    signals[i] = 0.25
                    position = 1
                elif curr_wr > -20:  # Deep overbought: look for short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought or opposite regime
            if curr_wr > -20 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold or opposite regime
            if curr_wr < -80 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals