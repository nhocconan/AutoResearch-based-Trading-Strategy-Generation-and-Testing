#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + 1w trend filter + volume confirmation
    # Williams %R identifies overextended moves (overbought/oversold)
    # 1w EMA provides major trend context - only trade with trend on higher timeframe
    # Volume spike confirms institutional participation in the reversal
    # Works in bull/bear by fading extremes in the direction of 1w trend
    # Target: 12-30 trades/year per symbol (60-150 over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA21 for trend filter
    ema_21_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 21:
        ema_21_1w[20] = np.mean(close_1w[:21])
        for i in range(21, len(df_1w)):
            ema_21_1w[i] = (close_1w[i] * 2 + ema_21_1w[i-1] * 19) / 21
    
    # Align 1w EMA21 to 6h timeframe (trend only changes at weekly close)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w, additional_delay_bars=0)
    
    # Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, 50.0)  # default neutral
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # avoid division by zero
    
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
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w trend: price above/below EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Williams %R extremes: oversold < -80, overbought > -20
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Entry logic: fade extremes in direction of 1w trend with volume confirmation
        long_entry = oversold and uptrend and volume_spike[i]
        short_entry = overbought and downtrend and volume_spike[i]
        
        # Exit logic: reverse signal or Williams %R returns to neutral zone
        long_exit = williams_r[i] > -50  # exited oversold zone
        short_exit = williams_r[i] < -50  # exited overbought zone
        
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

name = "6h_1w_williams_r_extreme_trend_v1"
timeframe = "6h"
leverage = 1.0