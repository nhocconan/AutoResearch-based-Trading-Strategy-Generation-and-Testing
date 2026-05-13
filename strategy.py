#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Long when Williams %R < -80 (oversold) and price > 1d EMA50, short when > -20 (overbought) and price < 1d EMA50.
# Uses discrete position sizing (0.25) to minimize fee drag. Targets 12-37 trades/year on 6h timeframe.
# Williams %R identifies exhaustion points; EMA50 ensures alignment with 1d trend; volume confirms conviction.
# Designed to work in both bull (buy oversold dips) and bear (sell overbought rallies) markets.

name = "6h_WilliamsR_MeanReversion_1dEMA50_Volume_Confirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    lookback_willr = 14
    highest_high = pd.Series(high).rolling(window=lookback_willr, min_periods=lookback_willr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_willr, min_periods=lookback_willr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_willr, lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i]) or
            (highest_high[i] == lowest_low[i])):  # Avoid division by zero
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) and price > 1d EMA50 and volume spike (>1.8x avg)
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) and price < 1d EMA50 and volume spike (>1.8x avg)
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to neutral (> -50) or volume drops
            if williams_r[i] > -50 or volume[i] < 0.6 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # EXIT SHORT: Williams %R returns to neutral (< -50) or volume drops
            if williams_r[i] < -50 or volume[i] < 0.6 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals