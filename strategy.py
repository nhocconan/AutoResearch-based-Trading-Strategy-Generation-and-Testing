#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme with 1d EMA(34) trend filter and volume confirmation (>1.5x 20-bar MA)
# Williams %R identifies overbought/oversold conditions. Extreme readings (< -80 or > -20) with trend alignment
# capture mean reversion in ranging markets and momentum in trending markets. Volume confirmation ensures
# institutional participation. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years.

name = "4h_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Williams %R (14-period) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 14, 34)  # Need 20 for volume MA, 14 for Williams %R, 34 for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), price above 1d EMA, and volume confirmation
            if curr_williams_r < -80.0 and curr_close > ema_1d_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below 1d EMA, and volume confirmation
            elif curr_williams_r > -20.0 and curr_close < ema_1d_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R rising above -50 (momentum weakening) or price below 1d EMA
            if curr_williams_r > -50.0 or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R falling below -50 (momentum weakening) or price above 1d EMA
            if curr_williams_r < -50.0 or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals