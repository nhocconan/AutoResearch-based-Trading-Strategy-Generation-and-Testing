#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; mean reversion works in ranging markets
# 1w EMA34 provides primary trend filter to avoid counter-trend trades
# Volume spike confirms participation
# Discrete sizing 0.25 to control risk and minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) to avoid overtrading
# Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend

name = "12h_WilliamsR_ME_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 1w EMA(34) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 30, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_williams_r = williams_r[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_close = close[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Williams %R oversold (< -80) AND price above 1w EMA34 (uptrend)
                if (curr_williams_r < -80 and 
                    curr_close > curr_ema_34_1w):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R overbought (> -20) AND price below 1w EMA34 (downtrend)
                elif (curr_williams_r > -20 and 
                      curr_close < curr_ema_34_1w):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns to neutral (> -50) or reverses
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns to neutral (< -50) or reverses
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals