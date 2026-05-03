#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA50 trend filter and volume confirmation
# Williams %R measures overbought/oversold: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# In bull markets: buy when %R crosses above -80 from below (oversold bounce) + price above 1d EMA50 + volume spike
# In bear markets: sell when %R crosses below -20 from above (overbought rejection) + price below 1d EMA50 + volume spike
# Works in both regimes by capturing mean reversion within the trend
# Volume spike (>1.8x 20-period EMA) confirms institutional participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "12h_WilliamsR_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 12h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Williams %R signals with 1d trend filter
        # Long: %R crosses above -80 from below + price above 1d EMA50 + volume spike
        # Short: %R crosses below -20 from above + price below 1d EMA50 + volume spike
        if position == 0:
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # Cross above -80
                close[i] > ema_50_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # Cross below -20
                  close[i] < ema_50_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: %R crosses below -50 (momentum loss) OR price below 1d EMA50
            if williams_r[i] < -50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: %R crosses above -50 (momentum loss) OR price above 1d EMA50
            if williams_r[i] > -50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals