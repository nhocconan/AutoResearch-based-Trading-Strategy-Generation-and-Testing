#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) extreme reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (< -80 oversold, > -20 overbought)
# 1d EMA34 filters trades to align with higher timeframe trend (long only above EMA, short only below)
# Volume confirmation (>1.3x 20-bar MA) ensures institutional participation
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR14_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Williams %R(14) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 1.3 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Williams %R and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(highest_high_14[i]) or 
            np.isnan(lowest_low_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Williams %R extreme levels
        oversold = curr_wr < -80
        overbought = curr_wr > -20
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold, price above 1d EMA34, volume spike
            if oversold and curr_close > ema_1d_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought, price below 1d EMA34, volume spike
            elif overbought and curr_close < ema_1d_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought or price below 1d EMA34
            if overbought or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold or price above 1d EMA34
            if oversold or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals