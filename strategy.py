#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w EMA trend filter for mean reversion in extreme conditions
# Williams %R identifies overbought/oversold conditions (> -20 = overbought, < -80 = oversold)
# 1w EMA provides trend context: only take mean reversion trades in direction of weekly trend
# Volume confirmation: current volume > 1.3x 20-period MA to validate the reversal
# Entry: Long when %R < -80 AND price > 1w EMA AND volume spike
# Entry: Short when %R > -20 AND price < 1w EMA AND volume spike
# Exit: When %R crosses back above -50 (for long) or below -50 (for short)
# Uses %R for extreme price points, weekly EMA for trend alignment, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_WilliamsR_1wEMA_TrendFilter_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate 1w EMA(34)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R(14) on 1d
    if len(high) >= 14:
        # Highest high over 14 periods
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        # Lowest low over 14 periods
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        # Williams %R = -100 * (highest_high - close) / (highest_high - lowest_low)
        williams_r = -100 * (highest_high - close) / np.where((highest_high - lowest_low) == 0, np.nan, (highest_high - lowest_low))
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.3 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND price above 1w EMA (uptrend) AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND price below 1w EMA (downtrend) AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (exiting oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (exiting overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals