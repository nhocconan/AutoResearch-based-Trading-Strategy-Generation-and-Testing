#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R with 1w trend filter and volume confirmation
    # Williams %R identifies overbought/oversold conditions on 6h
    # 1w EMA provides major trend direction (avoid counter-trend trades)
    # Volume spike confirms institutional participation at extremes
    # Works in bull/bear by fading extremes in range and following pullbacks in trend
    # Target: 12-37 trades/year per symbol (50-150 over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(20) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams %R(14) on 6h
    highest_high_14 = np.full(n, np.nan)
    lowest_low_14 = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(13, n):
        highest_high_14[i] = np.max(high[i-13:i+1])
        lowest_low_14[i] = np.min(low[i-13:i+1])
        if highest_high_14[i] != lowest_low_14[i]:
            williams_r[i] = (highest_high_14[i] - close[i]) / (highest_high_14[i] - lowest_low_14[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Williams %R extremes: < -80 = oversold, > -20 = overbought
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Entry logic: fade extremes with volume confirmation, but only with trend
        # In uptrend: buy oversold pullbacks
        # In downtrend: sell overbought bounces
        long_entry = oversold and volume_spike[i] and uptrend
        short_entry = overbought and volume_spike[i] and downtrend
        
        # Exit logic: return to neutral zone (%R between -50 and 50) or opposite extreme
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

name = "6h_1w_williams_r_trend_volume_v1"
timeframe = "6h"
leverage = 1.0