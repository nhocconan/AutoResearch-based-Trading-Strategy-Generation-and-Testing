#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian channels (20-period for structure)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 30-period average (strong filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Choppiness regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
    # Use 14-period chop on 1d timeframe
    true_range = np.maximum(high_1d - low_1d, np.maximum(abs(high_1d - np.roll(close_1d, 1)), abs(low_1d - np.roll(close_1d, 1))))
    true_range[0] = high_1d[0] - low_1d[0]  # first value
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Chop regime: > 61.8 = ranging (favor mean reversion), < 38.2 = trending (favor breakout)
        chop_value = chop_aligned[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # In ranging markets: mean reversion at Donchian channels
        # In trending markets: breakout continuation
        if is_ranging:
            # Mean reversion: sell at upper band, buy at lower band
            long_condition = (close[i] < lowest_low[i-1] and price_above_ema and volume_filter[i])
            short_condition = (close[i] > highest_high[i-1] and price_below_ema and volume_filter[i])
            # Exit when price returns to middle
            long_exit = (position == 1 and close[i] > (highest_high[i-1] + lowest_low[i-1]) / 2)
            short_exit = (position == -1 and close[i] < (highest_high[i-1] + lowest_low[i-1]) / 2)
        else:
            # Trending: breakout continuation
            long_condition = (close[i] > highest_high[i-1] and price_above_ema and volume_filter[i])
            short_condition = (close[i] < lowest_low[i-1] and price_below_ema and volume_filter[i])
            # Exit on opposite breakout
            long_exit = (position == 1 and close[i] < lowest_low[i-1])
            short_exit = (position == -1 and close[i] > highest_high[i-1])
        
        if long_condition:
            signals[i] = 0.25
            position = 1
        elif short_condition:
            signals[i] = -0.25
            position = -1
        elif position == 1 and long_exit:
            signals[i] = 0.0
            position = 0
        elif position == -1 and short_exit:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_ChopRegime_MeanRevert_Trend"
timeframe = "4h"
leverage = 1.0