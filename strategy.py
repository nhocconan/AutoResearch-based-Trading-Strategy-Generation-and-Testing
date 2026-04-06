#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter + volume confirmation
# Long when: price > Donchian upper (20) AND weekly EMA(50) trending up AND volume > 1.5x avg
# Short when: price < Donchian lower (20) AND weekly EMA(50) trending down AND volume > 1.5x avg
# Exit when: price crosses opposite Donchian band OR volume < 1.2x avg
# Uses daily timeframe for lower trade frequency, targets 30-100 total trades over 4 years
# Works in bull (breakouts work) and bear (short breakdowns) with trend filter

name = "1d_donchian_1w_ema_vol_v5"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Weekly EMA slope (trend direction)
    ema_slope = np.diff(weekly_ema_aligned, prepend=weekly_ema_aligned[0])
    ema_slope_pos = ema_slope > 0
    ema_slope_neg = ema_slope < 0
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    volume_low_threshold = 1.2 * volume_ma.values  # for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < donchian_low[i] or volume[i] < volume_low_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_up[i] or volume[i] < volume_low_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: price breaks above Donchian upper AND weekly EMA trending up AND volume confirmation
            if (close[i] > donchian_up[i] and ema_slope_pos[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND weekly EMA trending down AND volume confirmation
            elif (close[i] < donchian_low[i] and ema_slope_neg[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals