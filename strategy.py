#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA50 trend filter + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trendless markets when lines intertwine.
# Trades only when Alligator is "awake" (jaw > teeth > lips for long, reverse for short) aligned with 1w EMA50.
# Volume confirmation (1.5x 20-period EMA) filters low-participation breakouts.
# Designed for 30-100 total trades over 4 years (7-25/year) with discrete sizing to minimize fee drag.
# Works in both bull and bear markets by trading with the 1w trend only when Alligator confirms momentum.

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirmation"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d (Jaw=13, Teeth=8, Lips=5, all shifted)
    # Using typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, shifted 3 bars)
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines (already on 1d, no additional alignment needed for same timeframe)
    # But we need to ensure we're not using future data - the shifts above handle this
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start from 13 to have valid Alligator lines
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().iloc[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Alligator awake: jaw > teeth > lips for long, reverse for short
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # 1w trend filter: price above/below EMA50
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Alligator awake long + 1w uptrend + volume spike
            if alligator_long and uptrend_1w and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Alligator awake short + 1w downtrend + volume spike
            elif alligator_short and downtrend_1w and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses long alignment or 1w trend turns down
            if not (alligator_long and uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses short alignment or 1w trend turns up
            if not (alligator_short and downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals