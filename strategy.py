#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend + volume confirmation
# Williams %R identifies overbought/oversold conditions. In trending markets (price > 1d EMA34),
# we take mean reversion entries at extreme %R levels (-90 for long, -10 for short) with volume confirmation.
# In ranging markets, we fade at lesser extremes (-80 long, -20 short). Designed for low trade frequency
# (~12-37/year) to minimize fee drag. Works in bull/bear via regime adaptation.

name = "6h_WilliamsR_1dEMA34_RegimeAdaptive_VolumeConfirm_v2"
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
    
    # Load 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Regime filter: price above/below 1d EMA34 determines trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: mean reversion long at oversold (%R < -90) with volume
                if curr_wr < -90 and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend:
                # In downtrend: mean reversion short at overbought (%R > -10) with volume
                if curr_wr > -10 and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging market (near EMA): fade at lesser extremes
                if curr_wr < -80:
                    # Oversold: look for long
                    signals[i] = 0.25
                    position = 1
                elif curr_wr > -20:
                    # Overbought: look for short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when %R returns to neutral territory (> -50) or reverse signal
            if curr_wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when %R returns to neutral territory (< -50) or reverse signal
            if curr_wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals