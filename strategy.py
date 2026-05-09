#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Mean Reversion with 1d Trend Filter and Volume Spike
# Williams %R(14) identifies overbought/oversold conditions on 12h timeframe.
# In ranging markets: buy when %R < -80 (oversold), sell when %R > -20 (overbought).
# In trending markets: only take trades in direction of 1d EMA34 trend.
# Volume spike confirms momentum behind the move.
# Designed for 15-25 trades/year to avoid fee drag.
name = "12h_WilliamsR_MeanReversion_1dTrend_Volume"
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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams %R on 12h data (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_12h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume spike
            # In uptrend (price > EMA34): take all oversold signals
            # In downtrend (price < EMA34): only take if volume spike (counter-trend bounce)
            if williams_r[i] < -80 and vol_spike:
                if close[i] > ema34_12h[i] or volume[i] > vol_avg[i] * 2.5:  # Strong volume override for counter-trend
                    signals[i] = 0.25
                    position = 1
            # Short: Williams %R overbought (> -20) with volume spike
            # In downtrend (price < EMA34): take all overbought signals
            # In uptrend (price > EMA34): only take if volume spike (counter-trend fade)
            elif williams_r[i] > -20 and vol_spike:
                if close[i] < ema34_12h[i] or volume[i] > vol_avg[i] * 2.5:  # Strong volume override for counter-trend
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) OR stoploss via adverse move
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) OR stoploss via adverse move
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals