#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike (>2.0x 20-bar volume MA)
# Williams %R identifies overbought/oversold conditions; extremes (<-80 or >-20) signal potential reversals when aligned with 1d trend.
# Volume spike confirms strong participation at turning points. Works in ranging markets (mean reversion from extremes) and 
# trending markets (pullbacks to extreme in trend direction). Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR14_Extreme_1dEMA34_Trend_VolumeSpike_v1"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Williams %R (14-period) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 34  # Need 34 for 1d EMA and 14 for Williams %R (34 > 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Williams %R extreme conditions
        williams_oversold = williams_r[i] < -80  # Oversold
        williams_overbought = williams_r[i] > -20  # Overbought
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold, price above 1d EMA34, volume spike
            if williams_oversold and curr_close > ema_1d_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought, price below 1d EMA34, volume spike
            elif williams_overbought and curr_close < ema_1d_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought or price below 1d EMA34
            if williams_r[i] > -20 or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold or price above 1d EMA34
            if williams_r[i] < -80 or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals