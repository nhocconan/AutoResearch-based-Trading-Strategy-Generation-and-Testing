#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using weekly Donchian breakout with volume confirmation and ADX trend filter
# Weekly Donchian channels capture major trend structure. Breakouts above weekly high or below weekly low
# with volume > 1.8x 50-period average indicate strong momentum. ADX > 25 filters for trending markets.
# Works in bull/bear markets: breakouts capture trends, ADX filter avoids choppy whipsaws.
# Target: 20-50 total trades over 4 years (5-12/year) with 0.30 position sizing.

name = "4h_WeeklyDonchian55_VolumeADXFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels (55 periods) ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 56:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channel (55-period lookback)
    weekly_high = pd.Series(df_1w['high']).rolling(window=55, min_periods=55).max().values
    weekly_low = pd.Series(df_1w['low']).rolling(window=55, min_periods=55).min().values
    
    # Align weekly levels to 4h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: >1.8x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.8 * vol_ma_50)
    
    # ADX trend filter (14-period) on 4h timeframe
    # Calculate +DI, -DI, and ADX
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad to same length as original arrays
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    
    # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = np.zeros_like(tr)
    atr[14] = np.mean(tr[1:15])  # Initial ATR
    for i in range(15, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Directional Indicators
    plus_di = 100 * np.where(atr > 0, pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr, 0)
    minus_di = 100 * np.where(atr > 0, pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr, 0)
    dx = 100 * np.where((plus_di + minus_di) > 0, np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx = np.concatenate([np.full(27, np.nan), adx[27:]])  # Adjust for smoothing delay
    
    trend_filter = adx > 25
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly high with volume confirmation and trend
            if close[i] > weekly_high_aligned[i] and volume_filter[i] and trend_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short breakout: price breaks below weekly low with volume confirmation and trend
            elif close[i] < weekly_low_aligned[i] and volume_filter[i] and trend_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly low (trend reversal) or trailing stop
            if close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above weekly high (trend reversal) or trailing stop
            if close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals