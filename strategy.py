#!/usr/bin/env python3
name = "1d_Williams_Alligator_ElderRay_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Williams Alligator: 13, 8, 5 SMAs of median price (H+L)/2
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: EMA34 on weekly close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: jaws < teeth < lips for uptrend, reverse for downtrend
        alligator_long = jaw[i] < teeth[i] < lips[i]
        alligator_short = jaw[i] > teeth[i] > lips[i]
        
        # Elder Ray: bull power > 0 and bear power < 0 for strong trend
        strong_long = bull_power[i] > 0 and bear_power[i] < 0
        strong_short = bull_power[i] < 0 and bear_power[i] > 0
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema34_1d[i]
        weekly_downtrend = close[i] < ema34_1d[i]
        
        if position == 0:
            # Long: Alligator aligned up + strong bull power + weekly uptrend
            if alligator_long and strong_long and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + strong bear power + weekly downtrend
            elif alligator_short and strong_short and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator alignment breaks or weekly trend reverses
            if not alligator_long or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator alignment breaks or weekly trend reverses
            if not alligator_short or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals