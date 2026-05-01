#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume spike confirmation.
# Long when Bull Power > 0 AND 12h EMA50 rising AND volume > 1.8x 20-bar average.
# Short when Bear Power < 0 AND 12h EMA50 falling AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture medium-term trends with lower trade frequency.
# Elder Ray measures bull/bear strength relative to EMA13, filtering weak moves.
# 12h EMA50 ensures alignment with higher timeframe momentum.
# Volume spike requirement reduces false breakouts and improves signal quality.

name = "6h_ElderRay_12hEMA50_VolumeSpike_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 12h EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Calculate Elder Ray components: need EMA13 for Bull/Bear Power
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: current 6h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(ema_50_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # Elder Ray signals
        bull_strong = bull_power[i] > 0  # Bull Power positive
        bear_strong = bear_power[i] < 0  # Bear Power negative
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND 12h EMA50 rising AND volume confirmation
            if (bull_strong and 
                ema_50_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND 12h EMA50 falling AND volume confirmation
            elif (bear_strong and 
                  ema_50_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power turns negative OR 12h EMA50 falls (trend change)
            if (bear_power[i] >= 0 or 
                ema_50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR 12h EMA50 rises (trend change)
            if (bull_power[i] <= 0 or 
                ema_50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals