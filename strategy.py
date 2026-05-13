#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band and close > 1w EMA50 with volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band and close < 1w EMA50 with volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 30-100 total trades over 4 years on 1d timeframe.
# Donchian channels provide clear structure; 1w EMA50 filters counter-trend noise; volume confirms momentum.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm"
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
    
    # Calculate Donchian bands (20-period)
    lookback_dc = 20
    dc_upper = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_dc, lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band, close > 1w EMA50, volume spike
            if (high[i] > dc_upper[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band, close < 1w EMA50, volume spike
            elif (low[i] < dc_lower[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band OR volume drops below average
            if (low[i] < dc_lower[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band OR volume drops below average
            if (high[i] > dc_upper[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals