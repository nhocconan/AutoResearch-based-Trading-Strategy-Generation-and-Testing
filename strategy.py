#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) with 1w trend filter and volume spike
# Williams %R identifies overbought/oversold conditions. Buy when %R crosses above -80 from below (oversold bounce),
# sell when crosses below -20 from above (overbought reversal). 1w EMA50 ensures we trade with the weekly trend
# to avoid counter-trend whipsaws. Volume spike confirms institutional participation. This combines mean reversion
# with trend filtering, working in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Targets 12-30 trades per year (~50-120 total over 4 years) to minimize fee drag.

name = "12h_WilliamsR_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams %R (14) on 12h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # 1w EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume spike detection (24 periods = 12 days of 12h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, period)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from below (oversold bounce) 
            # AND price above weekly EMA50 (uptrend) AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_12h[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from above (overbought reversal)
            # AND price below weekly EMA50 (downtrend) AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_12h[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR trend weakens
            if williams_r[i] > -20 or close[i] < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR trend weakens
            if williams_r[i] < -80 or close[i] > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals