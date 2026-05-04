#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout with 1d Trend Filter and Volume Spike
# Camarilla pivots identify key intraday support/resistance levels. Breakout above R3 or below S3 with
# volume confirmation signals strong momentum. 1d EMA34 filter ensures alignment with higher timeframe trend.
# Designed for 12-37 trades/year on 12h to minimize fee drag. Works in bull markets via bullish breakouts
# above 1d EMA34 and in bear markets via bearish breakouts below 1d EMA34.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Need at least 100 bars of lookback for Camarilla calculation
        if i < 100:
            continue
            
        # Calculate Camarilla levels for prior day using lookback window
        # Look back 96 bars (4*24) to get prior day's OHLC for 12h timeframe
        lookback_start = max(0, i - 96)
        if lookback_start >= i:
            continue
            
        # Get prior day's OHLC (using 12h bars)
        prior_high = np.max(high[lookback_start:i])
        prior_low = np.min(low[lookback_start:i])
        prior_close = close[i-1]  # Previous bar's close
        
        # Calculate Camarilla levels
        range_val = prior_high - prior_low
        if range_val <= 0:
            continue
            
        # Camarilla R3 and S3 levels
        r3 = prior_close + range_val * 1.1 / 4
        s3 = prior_close - range_val * 1.1 / 4
        
        # Volume spike filter (20-period volume EMA)
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[:i+1].mean()
        volume_spike = volume[i] > (vol_ema_20 * 1.5)
        
        # Skip if HTF data not aligned yet
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND above 1d EMA34 AND volume spike
            if (close[i] > r3 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND below 1d EMA34 AND volume spike
            elif (close[i] < s3 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below R3 OR below 1d EMA34
            if close[i] < r3 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above S3 OR above 1d EMA34
            if close[i] > s3 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals