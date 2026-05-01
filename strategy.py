#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d Williams %R Reversal
# BB Width < 0.02 = squeeze (low volatility) → prepares for breakout
# Williams %R(14) on 1d: > -20 = overbought (short), < -80 = oversold (long)
# Entry: BB squeeze + Williams %R extreme + volume confirmation
# Exit: BB Width > 0.05 (high volatility) or Williams %R returns to neutral (-50)
# Works in bull/bear by trading reversals from extremes during low volatility periods
# Target: 12-35 trades/year via tight entry conditions (squeeze + extreme + volume)

name = "6h_BBWidth_Squeeze_WilliamsR_1d_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + (bb_std * std_20)
    bb_lower = sma_20 - (bb_std * std_20)
    bb_width = (bb_upper - bb_lower) / sma_20  # Normalized width
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(bb_period, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(bb_width[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        bb_squeeze = bb_width[i] < 0.02  # Low volatility squeeze
        bb_expansion = bb_width[i] > 0.05  # High volatility exit
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        williams_neutral = abs(williams_r_aligned[i] + 50) < 10  # Near -50
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze + Williams %R oversold + volume spike
            if bb_squeeze and williams_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze + Williams %R overbought + volume spike
            elif bb_squeeze and williams_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on BB expansion or Williams %R returns to neutral
            if bb_expansion or williams_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on BB expansion or Williams %R returns to neutral
            if bb_expansion or williams_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals