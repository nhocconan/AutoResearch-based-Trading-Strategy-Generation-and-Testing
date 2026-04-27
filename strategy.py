#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions. Long when %R crosses above -80 in uptrend (price > 1d EMA34).
# Short when %R crosses below -20 in downtrend (price < 1d EMA34). Volume > 1.5x average confirms momentum.
# Trend filter avoids counter-trend trades. Target: 15-25 trades/year to minimize fee decay.
# Williams %R is mean-reverting but works with trend filter to catch pullbacks in trends.
# Works in bull (buy pullbacks) and bear (sell rallies) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 12h data
    williams_period = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(williams_period, n):
        highest_high[i] = np.max(high[i-williams_period:i+1])
        lowest_low[i] = np.min(low[i-williams_period:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(williams_period, n):
        if highest_high[i] == lowest_low[i]:
            williams_r[i] = -50  # avoid division by zero
        else:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(williams_period, vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 (oversold) in uptrend with volume
            if i > start_idx and williams_r[i-1] <= -80 and williams_r[i] > -80 and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: Williams %R crosses below -20 (overbought) in downtrend with volume
            elif i > start_idx and williams_r[i-1] >= -20 and williams_r[i] < -20 and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R crosses below -50 or trend reverses
            if williams_r[i] < -50 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R crosses above -50 or trend reverses
            if williams_r[i] > -50 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsR_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0