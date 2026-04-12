#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
    # Williams %R < -80 = oversold (long), > -20 = overbought (short)
    # 1w EMA(34) determines trend direction to avoid counter-trend trades
    # Volume > 1.5x 24-period MA confirms momentum
    # Discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams %R (14-period) on 12h
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
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(24, n):
        if vol_ma_24[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_24[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA(34)
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Williams %R mean reversion conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Entry conditions with volume confirmation
        long_entry = oversold and (vol_ratio[i] > 1.5) and uptrend
        short_entry = overbought and (vol_ratio[i] > 1.5) and downtrend
        
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

name = "12h_1w_williams_r_mean_reversion_vol_v1"
timeframe = "12h"
leverage = 1.0