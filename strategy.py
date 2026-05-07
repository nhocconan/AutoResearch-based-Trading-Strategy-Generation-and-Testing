#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1-week EMA10 trend filter and volume spike.
# Long when price breaks above Donchian upper band AND price > 1w EMA10 with volume spike.
# Short when price breaks below Donchian lower band AND price < 1w EMA10 with volume spike.
# Uses 1w EMA10 trend filter to align with higher timeframe trend and avoid counter-trend trades.
# Volume spike filter ensures momentum confirmation. Designed for fewer trades (target: 7-25/year) to reduce fee drag.
# Works in both bull and bear markets by following the 1w trend direction.
name = "1d_Donchian20_1wEMA10_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w trend filter: 10-period EMA on close
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Donchian(20) channels from previous 1d bar to avoid lookahead
    # We calculate manually using rolling window with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1d volume average for spike detection
    vol_ema_1d = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_1d > 0, volume / vol_ema_1d, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1w EMA10
        uptrend = close[i] > ema_10_1w_aligned[i]
        downtrend = close[i] < ema_10_1w_aligned[i]
        
        if position == 0:
            # Long condition: break above Donchian upper band, in uptrend with volume spike
            long_condition = (close[i] > donchian_high[i]) and uptrend and vol_spike[i]
            # Short condition: break below Donchian lower band, in downtrend with volume spike
            short_condition = (close[i] < donchian_low[i]) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian lower band or trend turns down
            if (close[i] < donchian_low[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian upper band or trend turns up
            if (close[i] > donchian_high[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals