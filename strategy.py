#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Donchian breakouts capture strong momentum moves. The 1w EMA50 filter ensures we only
# trade in the direction of the weekly trend, reducing whipsaws in ranging markets.
# Volume confirmation adds conviction to breakouts. Designed to work in both bull and
# bear markets by filtering trades with the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1d Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        ema_trend = ema_50_aligned[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(upper_channel) or np.isnan(lower_channel):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Entry conditions: Donchian breakout in direction of 1w trend with volume spike
        if position == 0:
            # Long: price breaks above upper channel AND above 1w EMA50 (uptrend) with volume spike
            if close[i] > upper_channel and close[i] > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND below 1w EMA50 (downtrend) with volume spike
            elif close[i] < lower_channel and close[i] < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lower Donchian channel
            if close[i] < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian channel
            if close[i] > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals