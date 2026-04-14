#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian(10) breakout with 1-day EMA(21) trend filter and volume confirmation.
# The weekly Donchian captures longer-term breakouts, reducing trade frequency and avoiding noise.
# The daily EMA(21) ensures trades follow the intermediate trend, adapting to both bull and bear markets.
# Volume > 1.3x the 20-period average confirms institutional participation and reduces false breakouts.
# Exit occurs when price returns to the daily EMA(21) or breaks the opposite weekly Donchian band.
# This combination aims for 10-25 trades per year per symbol (40-100 total over 4 years), well within optimal range.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian channel (10 periods)
    dc_len = 10
    if len(df_1w) < dc_len:
        return np.zeros(n)
    
    dc_upper = pd.Series(df_1w['high']).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(df_1w['low']).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Daily EMA(21) for trend filter
    ema_len = 21
    ema_1d = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly Donchian to daily timeframe
    dc_upper_aligned = align_htf_to_ltf(prices, df_1w, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1w, dc_lower)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, dc_len, 20, ema_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper_aligned[i]) or 
            np.isnan(dc_lower_aligned[i]) or
            np.isnan(ema_1d[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to daily EMA21
        above_ema = close[i] > ema_1d[i]
        below_ema = close[i] < ema_1d[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Weekly Donchian breakout above + above daily EMA + volume
            if (close[i] > dc_upper_aligned[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Weekly Donchian breakdown below + below daily EMA + volume
            elif (close[i] < dc_lower_aligned[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to daily EMA or breaks below weekly Donchian lower
            if close[i] < ema_1d[i] or close[i] < dc_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to daily EMA or breaks above weekly Donchian upper
            if close[i] > ema_1d[i] or close[i] > dc_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian10_EMA21_Volume_v1"
timeframe = "1d"
leverage = 1.0