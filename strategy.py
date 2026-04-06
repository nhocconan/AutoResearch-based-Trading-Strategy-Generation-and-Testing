#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with weekly trend filter and volume confirmation.
# Long when Williams %R crosses above -50 (oversold recovery) during weekly uptrend with volume spike.
# Short when Williams %R crosses below -50 (overbought rejection) during weekly downtrend with volume spike.
# Uses weekly trend (price vs weekly SMA) to ensure trend alignment and volume spike for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "6h_williamsr_1w_trend_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14 periods)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Weekly trend filter: price vs weekly SMA(50)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_sma = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_uptrend = weekly_close > weekly_sma
    weekly_downtrend = weekly_close < weekly_sma
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or loss of trend/volume
        if position == 1:  # long position
            # Exit: Williams %R crosses below -50 or weekly turns downtrend or volume drops
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or \
               weekly_downtrend_aligned[i] or \
               not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -50 or weekly turns uptrend or volume drops
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or \
               weekly_uptrend_aligned[i] or \
               not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend filter and volume confirmation
            # Long: Williams %R crosses above -50 during weekly uptrend with volume spike
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) and \
               weekly_uptrend_aligned[i] and \
               volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50 during weekly downtrend with volume spike
            elif (williams_r[i] < -50 and williams_r[i-1] >= -50) and \
                 weekly_downtrend_aligned[i] and \
                 volume_spike[i]:
                signals[i] = -0.25
                position = -1
    
    return signals