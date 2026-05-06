#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian breakout with 1d trend filter and volume confirmation
# Daily Donchian(20) channels provide key breakout levels. Breakouts above upper band or below lower band
# with volume > 1.5x 20-period average and aligned with 1d EMA50 trend indicate strong momentum.
# Works in bull/bear markets: breakouts capture trends, EMA50 filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "12h_1dDonchian20_EMA50_Trend_Volume_v1"
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
    
    # Calculate 1d Donchian channels and EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's Donchian(20) - using prior completed day to avoid look-ahead
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1d EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume and uptrend confirmation
            if close[i] > donch_high_aligned[i] and volume_filter[i] and close[i] > ema50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low with volume and downtrend confirmation
            elif close[i] < donch_low_aligned[i] and volume_filter[i] and close[i] < ema50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low (failed breakout) or against trend
            if close[i] < donch_low_aligned[i] or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high (failed breakdown) or against trend
            if close[i] > donch_high_aligned[i] or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals