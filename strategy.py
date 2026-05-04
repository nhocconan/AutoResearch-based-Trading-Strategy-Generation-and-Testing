#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA34 Trend Filter + Volume Spike Confirmation
# Williams %R identifies overbought/oversold conditions in trending markets.
# Long when %R crosses above -80 (oversold recovery) in 1d uptrend with volume spike.
# Short when %R crosses below -20 (overbought rejection) in 1d downtrend with volume spike.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Designed for 20-50 trades/year on 4h to minimize fee drag while capturing mean reversions within trends.
# Works in bull markets via long signals in uptrend pullbacks and bear markets via short signals in downtrend rallies.

name = "4h_WilliamsR_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 4h data (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (oversold recovery) AND 1d uptrend AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # crossover above -80
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: Williams %R crosses below -20 (overbought rejection) AND 1d downtrend AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # crossover below -20
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR 1d trend turns down
            if (williams_r[i] > -20 and williams_r[i-1] <= -20) or \
               close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR 1d trend turns up
            if (williams_r[i] < -80 and williams_r[i-1] >= -80) or \
               close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals