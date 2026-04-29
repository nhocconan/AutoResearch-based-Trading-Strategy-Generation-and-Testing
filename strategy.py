#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA34 Trend Filter + Volume Spike
# Williams %R measures overbought/oversold: Long when %R < -80 (oversold) and rising,
# Short when %R > -20 (overbought) and falling.
# 1d EMA34 provides trend filter: only long in uptrend (price > EMA34), short in downtrend (price < EMA34).
# Volume spike confirms momentum: volume > 2.0x 20-period average.
# Discrete sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend filter.
# Timeframe: 12h (primary), HTF: 1d for EMA34 trend.

name = "12h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 20)  # warmup for EMA34, Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Williams %R rises above -50 (exiting oversold)
            # 2. Price falls below 1d EMA34 (trend change)
            if (curr_williams_r >= -50 or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Williams %R falls below -50 (exiting overbought)
            # 2. Price rises above 1d EMA34 (trend change)
            if (curr_williams_r <= -50 or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND rising AND price > 1d EMA34 AND volume spike
            if (curr_williams_r < -80 and
                i > start_idx and williams_r[i] > williams_r[i-1] and  # Williams %R rising
                curr_close > curr_ema_34_1d and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) AND falling AND price < 1d EMA34 AND volume spike
            elif (curr_williams_r > -20 and
                  i > start_idx and williams_r[i] < williams_r[i-1] and  # Williams %R falling
                  curr_close < curr_ema_34_1d and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals