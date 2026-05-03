#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1d EMA34 trend filter and volume confirmation
# The Williams Alligator (jaw/teeth/lips) identifies trend strength and direction.
# When lips cross above teeth and jaw with volume confirmation in uptrend → long
# When lips cross below teeth and jaw with volume confirmation in downtrend → short
# Uses 1d EMA34 for higher timeframe trend alignment to avoid counter-trend trades
# Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag
# Works in both bull and bear markets by trading with the 1d trend

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().shift(8).values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().shift(5).values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().shift(3).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: lips cross above teeth and jaw with volume confirmation in uptrend
            if (lips[i] > teeth[i] and lips[i] > jaw[i] and 
                lips[i-1] <= teeth[i-1] and lips[i-1] <= jaw[i-1] and
                is_uptrend and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: lips cross below teeth and jaw with volume confirmation in downtrend
            elif (lips[i] < teeth[i] and lips[i] < jaw[i] and 
                  lips[i-1] >= teeth[i-1] and lips[i-1] >= jaw[i-1] and
                  is_downtrend and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: lips cross below teeth (trend weakening)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: lips cross above teeth (trend weakening)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals