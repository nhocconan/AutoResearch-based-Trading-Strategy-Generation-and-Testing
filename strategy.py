#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_trend_v3
# Hypothesis: 12h Camarilla pivot breakout with volume confirmation (>1.5x 20-period average) and 1d HTF trend filter (price > 20-period EMA). Enters long when price breaks above H3 with volume and bullish trend; short when breaks below L3 with volume and bearish trend. Uses tighter volume filter (1.5x) and shorter EMA (20) to reduce trades and improve edge. Discrete sizing 0.25. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_trend_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # True range for the day
    day_range = prev_high - prev_low
    
    # Camarilla levels: H3 and L3 are key breakout levels
    H3 = prev_close + 1.0 * day_range
    L3 = prev_close - 1.0 * day_range
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1d HTF trend filter: 20-period EMA on 1d timeframe (shorter for faster adaptation)
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average (tighter filter)
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks L3 level
            if close[i] <= L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks H3 level
            if close[i] >= H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and 1d trend alignment
            if volume_confirmed:
                # Bullish 1d trend: price above 20-period EMA
                bullish_trend = close[i] > ema_20_1d_aligned[i]
                # Bearish 1d trend: price below 20-period EMA
                bearish_trend = close[i] < ema_20_1d_aligned[i]
                
                # Long: price breaks above H3 level with volume and bullish 1d trend
                if close[i] > H3_aligned[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below L3 level with volume and bearish 1d trend
                elif close[i] < L3_aligned[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals