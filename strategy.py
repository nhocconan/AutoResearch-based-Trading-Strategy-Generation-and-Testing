#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R combined with 1-week trend filter and volume confirmation.
Long when Williams %R crosses above -20 from oversold (< -80), weekly trend is up (price > 200 EMA), and volume > 1.5x average.
Short when Williams %R crosses below -80 from overbought (> -20), weekly trend is down (price < 200 EMA), and volume > 1.5x average.
Exit when Williams %R returns to the opposite extreme or weekly trend reverses.
Designed for low trade frequency (~15-30/year) to capture mean reversion within strong weekly trends.
Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
"""

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
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA200 for trend direction
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1w = close_1w > ema200_1w  # True if uptrend
    trend_down_1w = close_1w < ema200_1w  # True if downtrend
    
    # Align weekly trend to 6h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values
    
    # Volume average (20-period) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        trend_up = trend_up_aligned[i]
        trend_down = trend_down_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -20 from oversold (< -80), weekly uptrend, volume confirmation
            if (wr > -20 and wr_prev <= -20 and wr_prev < -80 and  # Cross above -20 from below -80
                trend_up and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from overbought (> -20), weekly downtrend, volume confirmation
            elif (wr < -80 and wr_prev >= -80 and wr_prev > -20 and  # Cross below -80 from above -20
                  trend_down and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to oversold (< -80) OR weekly trend turns down
                if wr < -80 or (trend_down and not trend_up):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to overbought (> -20) OR weekly trend turns up
                if wr > -20 or (trend_up and not trend_down):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0