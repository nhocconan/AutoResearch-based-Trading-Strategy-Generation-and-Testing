#!/usr/bin/env python3
"""
1d_Alligator_TRIX_Combo
Hypothesis: Bill Williams Alligator (Jaw/Teeth/Lips) defines trend direction and strength.
TRIX (15-period) filters for momentum confirmation and divergence. 
In trending markets: Alligator aligned (Lips > Teeth > Jaw for long, reverse for short) + TRIX rising/falling.
In ranging markets: Alligator intertwined (no clear order) + TRIX near zero → avoid trades.
Uses 1-week trend filter to avoid counter-trend trades. Low trade frequency expected due to strict alignment requirements.
Works in bull (follow green Alligator up) and bear (follow red Alligator down) by following Alligator alignment and TRIX momentum.
"""

name = "1d_Alligator_TRIX_Combo"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 50-period EMA on weekly for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator (13,8,5 smoothed with 8,5,3)
    # Jaw: 13-period SMMA smoothed 8 periods
    # Teeth: 8-period SMMA smoothed 5 periods  
    # Lips: 5-period SMMA smoothed 3 periods
    def smoothed_moving_average(arr, period):
        """SMMA: similar to EMA but with alpha = 1/period"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        res = np.full_like(arr, np.nan, dtype=float)
        alpha = 1.0 / period
        # First value is simple average
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (arr[i] * alpha) + (res[i-1] * (1 - alpha))
        return res
    
    jaw = smoothed_moving_average(close, 13)
    jaw = smoothed_moving_average(jaw, 8)  # additional smoothing
    teeth = smoothed_moving_average(close, 8)
    teeth = smoothed_moving_average(teeth, 5)
    lips = smoothed_moving_average(close, 5)
    lips = smoothed_moving_average(lips, 3)
    
    # TRIX: triple EMA + rate of change
    def ema_array(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema1 = ema_array(close, 15)
    ema2 = ema_array(ema1, 15)
    ema3 = ema_array(ema2, 15)
    trix = np.full_like(close, np.nan)
    # TRIX = 100 * (EMA3 today - EMA3 yesterday) / EMA3 yesterday
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # Align Alligator components to daily (already on daily, but ensure alignment)
    # Actually, Alligator is calculated on daily close, so no alignment needed for the values themselves
    # But we need to ensure we don't use incomplete data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Need enough for Alligator smoothing and weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(trix[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        # Jaw > Teeth > Lips = bearish alignment
        bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_aligned = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Weekly trend filter: price above/below weekly EMA50
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # TRIX momentum confirmation
        trix_rising = trix[i] > trix[i-1]
        trix_falling = trix[i] < trix[i-1]
        
        if position == 0:
            # Long: Alligator bullish aligned + weekly uptrend + TRIX rising
            if bullish_aligned and weekly_uptrend and trix_rising:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish aligned + weekly downtrend + TRIX falling
            elif bearish_aligned and weekly_downtrend and trix_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator loses alignment OR weekly trend turns down OR TRIX turns falling
            if not bullish_aligned or not weekly_uptrend or not trix_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses alignment OR weekly trend turns up OR TRIX turns rising
            if not bearish_aligned or not weekly_downtrend or not trix_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals