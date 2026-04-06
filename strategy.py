#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(20) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND price > 1d EMA(20) AND volume > 2x average
# Short when price breaks below 4h Donchian lower band AND price < 1d EMA(20) AND volume > 2x average
# Exit when price returns to Donchian midline (average of upper/lower) or volume < 1x average
# Uses 4h timeframe with 1d trend filter to reduce false breakouts and improve win rate
# Targets 75-200 total trades over 4 years (19-50/year) with low frequency to minimize fee drag
# Works in both bull and bear markets by requiring trend alignment (EMA filter) and volume confirmation

name = "4h_donchian_20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # 1d EMA(20) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate EMA(20) on daily close
    ema_20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean()
    ema_20 = ema_20.values
    
    # Align daily EMA to 4h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    volume_low_threshold = 1.0 * volume_ma.values  # Exit when volume drops below average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to Donchian midline OR volume drops below average
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or volume[i] < volume_low_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or volume[i] < volume_low_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, EMA trend filter, and volume confirmation
            # Long: price breaks above upper band AND price > EMA(20) AND volume > 2x average
            if (close[i] > donchian_upper[i] and close[i] > ema_20_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND price < EMA(20) AND volume > 2x average
            elif (close[i] < donchian_lower[i] and close[i] < ema_20_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals