#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 12h trend filter and volume confirmation
# Williams %R measures overbought/oversold levels: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) and 12h EMA50 is rising (uptrend)
# Short when %R > -20 (overbought) and 12h EMA50 is falling (downtrend)
# Volume confirmation: current volume > 1.5 * 20-period average volume to filter weak moves
# Uses discrete sizing (0.25) to minimize fee churn and manage drawdowns
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # 12h HTF data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) on 12h close
    ema_12h_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Williams %R (14-period) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # 12h EMA trend: rising if current > previous, falling if current < previous
        ema_rising = ema_12h_50_aligned[i] > ema_12h_50_aligned[i-1]
        ema_falling = ema_12h_50_aligned[i] < ema_12h_50_aligned[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold), 12h EMA50 rising, volume spike
            if curr_williams_r < -80 and ema_rising and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), 12h EMA50 falling, volume spike
            elif curr_williams_r > -20 and ema_falling and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R > -50 (exiting oversold) or 12h EMA50 falling
            if curr_williams_r > -50 or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R < -50 (exiting overbought) or 12h EMA50 rising
            if curr_williams_r < -50 or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals