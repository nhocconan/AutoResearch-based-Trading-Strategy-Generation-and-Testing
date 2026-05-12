#!/usr/bin/env python3
"""
12h_1d_Volume_Weighted_VWAP_Breakout_v1
Hypothesis: Breakouts from volume-weighted VWAP deviations on 12h timeframe with 1-day trend filter.
Uses VWAP deviation bands (similar to Bollinger but volume-weighted) to identify overextended moves,
then fades them when price reverts toward VWAP with volume confirmation.
Works in both bull and bear markets: fades overextended moves in ranging markets,
and follows volume-confirmed breakouts in trending markets.
"""

name = "12h_1d_Volume_Weighted_VWAP_Breakout_v1"
timeframe = "12h"
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
    
    # VWAP calculation (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative VWAP reset each period - using 50-period window
    vwap_num = pd.Series(vwap_numerator).rolling(window=50, min_periods=50).sum().values
    vwap_den = pd.Series(vwap_denominator).rolling(window=50, min_periods=50).sum().values
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # VWAP deviation bands (volume-weighted standard deviation)
    vwap_diff = typical_price - vwap
    vwap_var = (vwap_diff * vwap_diff) * volume
    vwap_var_sum = pd.Series(vwap_var).rolling(window=50, min_periods=50).sum().values
    vwap_vol_sum = pd.Series(volume).rolling(window=50, min_periods=50).sum().values
    vwap_variance = np.divide(vwap_var_sum, vwap_vol_sum, out=np.full_like(vwap_var_sum, np.nan), where=vwap_vol_sum!=0)
    vwap_std = np.sqrt(np.maximum(vwap_variance, 0))
    
    # Upper and lower bands (2 standard deviations)
    vwap_upper = vwap + (2.0 * vwap_std)
    vwap_lower = vwap - (2.0 * vwap_std)
    
    # Volume spike: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(vwap[i]) or
            np.isnan(vwap_upper[i]) or
            np.isnan(vwap_lower[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above VWAP lower band with volume spike AND price below 1d EMA50 (fade overextension)
            # OR price crosses above VWAP with volume spike AND price above 1d EMA50 (follow breakout)
            if ((close[i] > vwap_lower[i] and close[i-1] <= vwap_lower[i-1]) or \
                (close[i] > vwap[i] and close[i-1] <= vwap[i-1])) and \
               volume_spike[i]:
                # Determine if fading or following based on trend
                if close[i] < ema_50_1d_aligned[i]:
                    # Fading oversold condition in downtrend
                    signals[i] = 0.25
                    position = 1
                else:
                    # Following breakout in uptrend
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price crosses below VWAP upper band with volume spike AND price above 1d EMA50 (fade overextension)
            # OR price crosses below VWAP with volume spike AND price below 1d EMA50 (follow breakout)
            elif ((close[i] < vwap_upper[i] and close[i-1] >= vwap_upper[i-1]) or \
                  (close[i] < vwap[i] and close[i-1] >= vwap[i-1])) and \
                 volume_spike[i]:
                if close[i] > ema_50_1d_aligned[i]:
                    # Fading overbought condition in uptrend
                    signals[i] = -0.25
                    position = -1
                else:
                    # Following breakdown in downtrend
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches VWAP or VWAP upper band
            if close[i] >= vwap[i] or close[i] >= vwap_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches VWAP or VWAP lower band
            if close[i] <= vwap[i] or close[i] <= vwap_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals