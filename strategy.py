#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme with 1w trend filter and volume spike confirmation
    # Williams %R < -90 = deeply oversold (long), > -10 = deeply overbought (short)
    # 1w EMA(50) determines primary trend to avoid counter-trend trades in bear markets
    # Volume > 2.0x 20-period MA confirms genuine momentum (not chop)
    # Discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R (14-period) on 6h
    highest_14 = np.full(n, np.nan)
    lowest_14 = np.full(n, np.nan)
    
    for i in range(14, n):
        highest_14[i] = np.max(high[i-14:i])
        lowest_14[i] = np.min(low[i-14:i])
    
    williams_r = np.full(n, np.nan)
    for i in range(14, n):
        if highest_14[i] != lowest_14[i]:
            williams_r[i] = (highest_14[i] - close[i]) / (highest_14[i] - lowest_14[i]) * -100
        else:
            williams_r[i] = -50.0
    
    # Volume confirmation: current volume > 2.0x 20-period average (strict for low frequency)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA(50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Williams %R extreme conditions (deep oversold/overbought)
        deeply_oversold = williams_r[i] < -90
        deeply_overbought = williams_r[i] > -10
        
        # Entry conditions with strict volume confirmation
        long_entry = deeply_oversold and (vol_ratio[i] > 2.0) and uptrend
        short_entry = deeply_overbought and (vol_ratio[i] > 2.0) and downtrend
        
        # Exit conditions: Williams %R returns to midpoint (-50)
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_williams_r_extreme_vol_v1"
timeframe = "6h"
leverage = 1.0