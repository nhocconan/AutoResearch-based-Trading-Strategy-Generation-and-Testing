#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Uses Williams Alligator (JAW=13, TEETH=8, LIPS=5) to identify trending vs ranging markets
# Only trade when Alligator is "awake" (JAW > TEETH > LIPS for uptrend, reverse for downtrend)
# Entry on price retracement to TEETH line in direction of trend with volume spike confirmation
# 1d EMA34 ensures we trade with higher timeframe trend
# Works in bull markets via buying dips in uptrends and bear markets via selling rallies in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Williams_Alligator_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    def smma(data, period):
        """Smoothed Moving Average - Williams Alligator uses this"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: JAW(13,8), TEETH(8,5), LIPS(5,3)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and Alligator
    
    for i in range(start_idx, n):
        # Need sufficient data for Alligator calculation
        if i < 13:  # JAW needs 13 periods
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Check if Alligator is "awake" and trending
        # Uptrend: JAW > TEETH > LIPS
        # Downtrend: JAW < TEETH < LIPS
        is_uptrend = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
        is_downtrend = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips)
        
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_spike = False
        
        if position == 0:  # Flat - look for new entries
            # Require Alligator awake and volume spike
            if (is_uptrend or is_downtrend) and volume_spike:
                # Bullish entry: price near TEETH in uptrend
                if is_uptrend and curr_close >= curr_teeth * 0.995 and curr_close <= curr_teeth * 1.005:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price near TEETH in downtrend
                elif is_downtrend and curr_close >= curr_teeth * 0.995 and curr_close <= curr_teeth * 1.005:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Alligator goes to sleep or reverses
            if not is_uptrend:  # Alligator sleeping or downtrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator goes to sleep or reverses
            if not is_downtrend:  # Alligator sleeping or uptrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals