#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume confirmation
# Williams %R measures overbought/oversold: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) AND 1d EMA34 trending up (close > EMA34) AND volume > 1.5x 20 EMA
# Short when %R > -20 (overbought) AND 1d EMA34 trending down (close < EMA34) AND volume > 1.5x 20 EMA
# Uses 6h timeframe for lower frequency, Williams %R for mean reversion in extremes, 1d EMA34 for trend filter,
# volume confirmation to avoid false signals. Designed for 12-37 trades/year with discrete sizing (0.25).
# Works in bull markets via buying dips in uptrends and bear markets via selling rallies in downtrends.

name = "6h_WilliamsR_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Trend up: close > EMA34, Trend down: close < EMA34
    ema_trend_up = close_1d > ema_34_1d
    ema_trend_down = close_1d < ema_34_1d
    
    # Align 1d EMA trends to 6h timeframe
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_up.astype(float))
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_down.astype(float))
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 1d EMA trend up AND volume spike
            if (williams_r[i] < -80 and 
                ema_trend_up_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 1d EMA trend down AND volume spike
            elif (williams_r[i] > -20 and 
                  ema_trend_down_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (recovered from oversold) OR 1d trend changes
            if (williams_r[i] > -50 or 
                ema_trend_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (recovered from overbought) OR 1d trend changes
            if (williams_r[i] < -50 or 
                ema_trend_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals