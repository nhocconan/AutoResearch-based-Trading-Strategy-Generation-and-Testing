#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper AND close > 1w EMA34 AND volume > 2.0x average
# Short when price breaks below Donchian lower AND close < 1w EMA34 AND volume > 2.0x average
# Exit when price crosses Donchian middle (mean reversion) OR trend reversal (price crosses 1w EMA34)
# Uses 1d timeframe for lower frequency, Donchian for structure, 1w EMA for HTF trend filter, volume spike for confirmation.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull via breakout continuation, bear via faded rallies.

name = "1d_Donchian20_1wEMA34_Volume_v1"
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
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) on 1d data (using previous 20 bars)
    if len(high_1d) >= 20:
        upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
        lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
        middle_1d = (upper_1d + lower_1d) / 2
    else:
        upper_1d = np.full_like(high_1d, np.nan)
        lower_1d = np.full_like(low_1d, np.nan)
        middle_1d = np.full_like(high_1d, np.nan)
    
    # Align Donchian levels to 1d timeframe (already aligned since calculated on 1d)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_1d)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current 1d volume > 2.0x 20-period average (spike confirmation)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter = align_htf_to_ltf(prices, df_1d, vol_ma_1d * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for Donchian and EMA
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper AND close > 1w EMA34 AND volume spike
            if close[i] > upper_aligned[i] and close[i] > ema34_1w_aligned[i] and volume[i] > volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower AND close < 1w EMA34 AND volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema34_1w_aligned[i] and volume[i] > volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < middle (mean reversion) OR trend reversal (close < 1w EMA34)
            if close[i] < middle_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > middle (mean reversion) OR trend reversal (close > 1w EMA34)
            if close[i] > middle_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals