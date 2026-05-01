#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (>80 or <20) 
# combined with volume spike and daily trend alignment provide high-probability mean reversion 
# entries in ranging markets and breakout failures in trending markets.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull (fade breakouts at extremes) and bear (mean reversion in range).

name = "12h_WilliamsR14_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
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
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Williams %R calculation on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for 1d EMA, 14 for Williams %R, 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), price above 1d EMA50, volume spike
            if curr_williams_r < -80 and curr_close > ema_1d_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below 1d EMA50, volume spike
            elif curr_williams_r > -20 and curr_close < ema_1d_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R crossing above -50 or price below 1d EMA50
            if curr_williams_r > -50 or curr_close < ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R crossing below -50 or price above 1d EMA50
            if curr_williams_r < -50 or curr_close > ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals