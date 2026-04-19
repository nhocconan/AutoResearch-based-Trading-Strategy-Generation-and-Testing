#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d volume spike + 1w trend filter
# - Williams %R(14) identifies overbought/oversold conditions
# - Entry when %R crosses above -50 in uptrend or below -50 in downtrend
# - 1d volume > 2.0x 20-period average for conviction
# - 1w EMA(50) defines long-term trend (only trade in direction of weekly trend)
# - Exit when %R reaches opposite extreme (-20 for long, -80 for short)
# - Position size: 0.25 to balance risk and return
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_WilliamsR_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 1d average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 1w EMA50) + Williams %R crosses above -50 + volume
            if (close[i] > ema_50_1w_aligned[i] and 
                williams_r[i] > -50 and williams_r[i-1] <= -50 and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1w EMA50) + Williams %R crosses below -50 + volume
            elif (close[i] < ema_50_1w_aligned[i] and 
                  williams_r[i] < -50 and williams_r[i-1] >= -50 and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when Williams %R reaches overbought (-20) or trend reversal
            if williams_r[i] >= -20 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when Williams %R reaches oversold (-80) or trend reversal
            if williams_r[i] <= -80 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals