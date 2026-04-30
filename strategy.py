#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 12h EMA34 trend filter and volume confirmation
# Williams %R measures overbought/oversold conditions. We use 12h EMA34 as trend filter 
# (bullish when price > EMA34, bearish when price < EMA34). Volume spike (2.0x 20-period average) 
# confirms mean reversion strength. Discrete sizing 0.25 minimizes fee churn. Works in bull via 
# oversold longs with uptrend, in bear via overbought shorts with downtrend. Target: 12-37 trades/year 
# (50-150 total over 4 years) by using strict Williams %R thresholds (-80 for oversold, -20 for overbought).

name = "6h_WilliamsR_ME_12hEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14, 34)  # warmup for volume MA, Williams %R, and 12h EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or np.isnan(williams_r[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Williams %R < -80 (oversold) AND price > 12h EMA34 (uptrend)
                if curr_williams_r < -80 and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R > -20 (overbought) AND price < 12h EMA34 (downtrend)
                elif curr_williams_r > -20 and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -50 (mean reversion complete) OR price drops below 6h EMA13
            ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
            if curr_williams_r > -50 or curr_close < ema_13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (mean reversion complete) OR price rises above 6h EMA13
            ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
            if curr_williams_r < -50 or curr_close > ema_13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals