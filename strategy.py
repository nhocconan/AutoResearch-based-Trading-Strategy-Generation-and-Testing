#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions on daily timeframe
# 1w EMA50 provides major trend direction to avoid counter-trend trades
# Volume confirmation ensures strong participation at reversal points
# Designed to work in both bull and bear markets by fading extremes in the direction of the weekly trend
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag on 1d timeframe

name = "1d_WilliamsR_MeanReversion_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R on 1d data (14-period)
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams %R mean reversion with 1w trend filter
        # Long: Williams %R < -80 (oversold) + volume spike + price above 1w EMA50 (uptrend)
        # Short: Williams %R > -20 (overbought) + volume spike + price below 1w EMA50 (downtrend)
        if position == 0:
            if (williams_r[i] < -80 and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (williams_r[i] > -20 and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (return to neutral) OR price below 1w EMA50 (trend change)
            if williams_r[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (return to neutral) OR price above 1w EMA50 (trend change)
            if williams_r[i] < -50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals