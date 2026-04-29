#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Uses 12h primary timeframe targeting 50-150 trades over 4 years (12-37/year)
# Long: price breaks above R3 with 1w EMA34 uptrend and volume spike
# Short: price breaks below S3 with 1w EMA34 downtrend and volume spike
# Exit: price reverts to Camarilla pivot point (PP) or trend reversal
# Designed for both bull and bear markets via trend filter and mean reversion exit

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    PP = (high_1d + low_1d + close_1d) / 3
    R3 = PP + (high_1d - low_1d) * 1.1 / 2
    S3 = PP - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # max(20 for vol, 34 for EMA) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(PP_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_PP = PP_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price below pivot point OR trend reversal (price below EMA)
            if curr_close < curr_PP or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above pivot point OR trend reversal (price above EMA)
            if curr_close > curr_PP or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND uptrend (price above EMA) AND volume spike
            if (curr_close > curr_R3 and 
                curr_close > curr_ema_1w and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND downtrend (price below EMA) AND volume spike
            elif (curr_close < curr_S3 and 
                  curr_close < curr_ema_1w and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals