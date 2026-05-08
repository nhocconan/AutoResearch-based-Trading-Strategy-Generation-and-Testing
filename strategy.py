#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Liquidity_Grab_Fade_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    # Get 1d data for trend and liquidity levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's high and low for liquidity grab detection
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks below previous day's low (liquidity grab) then reverses above it,
            # with uptrend bias (price above 1d EMA50) and volume confirmation
            liq_grab_long = low[i] < prev_low_aligned[i] and close[i] > prev_low_aligned[i]
            long_cond = liq_grab_long and (close[i] > ema_50_1d_aligned[i]) and volume_filter[i]
            
            # Short: price breaks above previous day's high (liquidity grab) then reverses below it,
            # with downtrend bias (price below 1d EMA50) and volume confirmation
            liq_grab_short = high[i] > prev_high_aligned[i] and close[i] < prev_high_aligned[i]
            short_cond = liq_grab_short and (close[i] < ema_50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below previous day's low (stop loss) or
            # takes profit at previous day's high
            if low[i] < prev_low_aligned[i]:  # stop loss
                signals[i] = 0.0
                position = 0
            elif high[i] >= prev_high_aligned[i]:  # take profit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above previous day's high (stop loss) or
            # takes profit at previous day's low
            if high[i] > prev_high_aligned[i]:  # stop loss
                signals[i] = 0.0
                position = 0
            elif low[i] <= prev_low_aligned[i]:  # take profit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals