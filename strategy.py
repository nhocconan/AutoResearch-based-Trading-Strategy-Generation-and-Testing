#!/usr/bin/env python3
"""
12h_1w_Wick_Pattern_Reversal_v1
Hypothesis: Price rejection at weekly support/resistance triggers reversals in ranging markets. 
Long when price closes above weekly low with bullish engulfing candle and volume spike.
Short when price closes below weekly high with bearish engulfing candle and volume spike.
Use 1-day ADX < 20 to filter for ranging markets and avoid false signals in trends.
Works in bull markets (buy dips at weekly support) and bear markets (sell rallies at weekly resistance).
Target: 25-35 trades/year with low frequency and high win rate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Wick_Pattern_Reversal_v1"
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
    open_price = prices['open'].values
    
    # === WEEKLY DATA FOR SUPPORT/RESISTANCE LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly high and low (support/resistance levels)
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Align weekly levels to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === DAILY DATA FOR RANGE FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) for ranging market filter
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12H PATTERN DETECTION ===
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulfing = (close > open_price) & (open_price <= np.roll(close, 1)) & (close >= np.roll(open_price, 1)) & (np.roll(close, 1) < np.roll(open_price, 1))
    
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulfing = (close < open_price) & (open_price >= np.roll(close, 1)) & (close <= np.roll(open_price, 1)) & (np.roll(close, 1) > np.roll(open_price, 1))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Range filter: ADX < 20 indicates ranging market (avoid trends)
        ranging = adx_aligned[i] < 20
        
        # Long setup: price near weekly low + bullish engulfing + volume spike
        near_weekly_low = close[i] <= weekly_low_aligned[i] * 1.01  # within 1% above weekly low
        long_signal = near_weekly_low and bullish_engulfing[i] and volume_spike[i] and ranging
        
        # Short setup: price near weekly high + bearish engulfing + volume spike
        near_weekly_high = close[i] >= weekly_high_aligned[i] * 0.99  # within 1% below weekly high
        short_signal = near_weekly_high and bearish_engulfing[i] and volume_spike[i] and ranging
        
        # Exit: price moves to middle of weekly range or trend emerges
        weekly_mid = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2
        exit_long = (position == 1 and (close[i] >= weekly_mid or adx_aligned[i] >= 25))
        exit_short = (position == -1 and (close[i] <= weekly_mid or adx_aligned[i] >= 25))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals