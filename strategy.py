#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation
# Long when price breaks above Donchian upper band (20) AND price > 1w EMA AND volume > 1.5x average
# Short when price breaks below Donchian lower band (20) AND price < 1w EMA AND volume > 1.5x average
# Exit when price returns to Donchian middle (10-day average of high/low) OR opposite breakout
# Uses 1d timeframe to reduce trade frequency, targets 30-100 total trades over 4 years
# Works in both bull/bear markets by following trend (1w EMA) with volatility-based entries

name = "1d_donchian_1w_ema_vol_v6"
timeframe = "1d"
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
    
    # Donchian Channel (20-period) - volatility breakout
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = ((highest_high + lowest_low) / 2).values
    
    # 1-week EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_1w = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to middle OR opposite breakout
        if position == 1:  # long position
            if close[i] <= donchian_middle[i] or low[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_middle[i] or high[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts in direction of 1w EMA trend with volume confirmation
            # Long: price breaks above upper band AND above 1w EMA AND volume confirmation
            if (high[i] > donchian_upper[i] and close[i] > ema_1w_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND below 1w EMA AND volume confirmation
            elif (low[i] < donchian_lower[i] and close[i] < ema_1w_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals