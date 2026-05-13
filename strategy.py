#!/usr/bin/env python3
# Hypothesis: 1d Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; 1w EMA50 provides trend direction;
# volume spike confirms momentum. Designed for low trade frequency (<50 total 1d trades)
# to minimize fee drag while capturing reversals in both bull and bear markets.

name = "1d_WilliamsR_MeanReversion_1wEMA50_Volume_Confirm_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period)
    lookback_willr = 14
    highest_high = pd.Series(high).rolling(window=lookback_willr, min_periods=lookback_willr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_willr, min_periods=lookback_willr).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_willr, lookback_vol) + 1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(willr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price above 1w EMA50, volume spike (>1.5x avg)
            if (willr[i] < -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price below 1w EMA50, volume spike (>1.5x avg)
            elif (willr[i] > -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R rises above -50 (mean reversion) or volume drops
            if (willr[i] > -50) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # EXIT SHORT: Williams %R falls below -50 (mean reversion) or volume drops
            if (willr[i] < -50) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals