#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Confirmation
# Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought
# Enter long when %R crosses above -80 from below (end of oversold) with 1d uptrend (close > EMA34)
# Enter short when %R crosses below -20 from above (end of overbought) with 1d downtrend (close < EMA34)
# Volume spike (2.0x 20-period average) confirms institutional participation at reversal
# Works in bull markets via oversold bounces in uptrend and in bear markets via overbought reversals in downtrend
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe
# Discrete position sizing (0.25) minimizes fee churn

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, EMA, and volume MA)
    start_idx = 34  # buffer for 34-period EMA and 14-period Williams %R
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or i == 0):
            signals[i] = 0.0
            continue
        
        # Williams %R crossover signals
        williams_r_prev = williams_r[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (end of oversold) + 1d close > EMA34 + volume spike
            if (williams_r_prev <= -80 and williams_r[i] > -80 and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (end of overbought) + 1d close < EMA34 + volume spike
            elif (williams_r_prev >= -20 and williams_r[i] < -20 and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) or 1d trend breaks
            if (williams_r_prev < -20 and williams_r[i] >= -20) or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) or 1d trend breaks
            if (williams_r_prev > -80 and williams_r[i] <= -80) or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals