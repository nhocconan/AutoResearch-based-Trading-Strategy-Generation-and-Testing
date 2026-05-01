#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions. Extreme readings (<-90 or >-10) 
# signal exhaustion and potential reversal. 1d EMA34 ensures alignment with daily trend.
# Volume spike confirms institutional participation in the reversal. Works in bull markets 
# (buying dips in uptrend) and bear markets (selling rallies in downtrend). Discrete sizing 
# (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) calculation
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 6h timeframe
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 34 for EMA + 20 for volume MA + buffer
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = curr_close > ema_34_1d_aligned[i]
        trend_down = curr_close < ema_34_1d_aligned[i]
        
        # Williams %R extreme conditions
        wr_oversold = williams_r[i] < -90  # Extremely oversold
        wr_overbought = williams_r[i] > -10  # Extremely overbought
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R extremely oversold, volume spike, uptrend
            if wr_oversold and vol_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R extremely overbought, volume spike, downtrend
            elif wr_overbought and vol_spike and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R recovery or trend reversal
            if williams_r[i] > -50 or not trend_up:  # Exit when no longer oversold or trend breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R recovery or trend reversal
            if williams_r[i] < -50 or not trend_down:  # Exit when no longer overbought or trend breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals