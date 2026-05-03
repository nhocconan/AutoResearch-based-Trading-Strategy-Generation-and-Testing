#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R reversal + 1w EMA34 trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extremes capture mean reversion
# Works in bull/bear: 1w EMA34 ensures we trade with higher timeframe trend to avoid whipsaws
# Volume spike (>2.0x 20-period EMA) confirms reversal authenticity
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# Williams %R is effective in ranging markets and during trend pullbacks

name = "1d_WilliamsR_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R on 1d data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams %R reversal signals with 1w trend filter
        # Long: %R crosses above -80 from oversold + price above 1w EMA34 + volume spike
        # Short: %R crosses below -20 from overbought + price below 1w EMA34 + volume spike
        if position == 0:
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_34_1w_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_34_1w_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: %R crosses below -50 (momentum loss) OR price below 1w EMA34
            if williams_r[i] < -50 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: %R crosses above -50 (momentum loss) OR price above 1w EMA34
            if williams_r[i] > -50 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals