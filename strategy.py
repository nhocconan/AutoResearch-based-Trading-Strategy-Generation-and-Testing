#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme with 1w trend filter and volume confirmation
# Uses Williams %R(14) from daily data for extreme overbought/oversold signals
# 1w EMA(34) provides strong trend filter to avoid counter-trend whipsaws
# Volume spike (2.0x 20-period average) ensures participation and reduces false signals
# Only takes reversals from extremes in the direction of the 1w trend
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend
# Williams %R captures exhaustion moves that often precede reversals in ranging/ bear markets

name = "1d_WilliamsR_Extreme_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R(14) from 1d data
    # Need at least 14 periods for Williams %R
    if len(close) < 15:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, EMA and volume MA)
    start_idx = max(20, 34) + 5  # max(volume MA 20, Williams %R 14, EMA 34) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R below -80 (oversold) AND uptrend AND volume confirm
            if (williams_r[i] < -80 and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R above -20 (overbought) AND downtrend AND volume confirm
            elif (williams_r[i] > -20 and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R above -20 (overbought) OR trend reverses to downtrend
            if (williams_r[i] > -20 or 
                not uptrend):  # exited if price closes below 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R below -80 (oversold) OR trend reverses to uptrend
            if (williams_r[i] < -80 or 
                not downtrend):  # exited if price closes above 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals