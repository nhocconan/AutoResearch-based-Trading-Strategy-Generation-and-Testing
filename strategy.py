#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with weekly trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; weekly trend ensures directional bias
# Volume confirms breakout strength. Designed for mean reversion in ranging markets and
# trend continuation in trending markets, suitable for both bull and bear regimes.
# Target: 15-30 trades/year to stay within fee limits.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Weekly trend: EMA 34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 12h volume: ratio of current volume to 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # 12h price
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend bias
        bullish_trend = close[i] > ema_34_1w_aligned[i]
        bearish_trend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation: above average volume
        vol_confirm = vol_ratio[i] > 1.5
        
        if position == 0 and vol_confirm:
            # Long: oversold in bullish trend or extreme oversold
            if (bullish_trend and williams_r_aligned[i] < -80) or williams_r_aligned[i] < -90:
                signals[i] = 0.25
                position = 1
            # Short: overbought in bearish trend or extreme overbought
            elif (bearish_trend and williams_r_aligned[i] > -20) or williams_r_aligned[i] > -10:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: overbought or trend change
            if williams_r_aligned[i] > -20 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: oversold or trend change
            if williams_r_aligned[i] < -80 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_WeeklyTrend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0