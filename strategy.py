#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian breakout with volume confirmation and 1d EMA trend filter
# Breakout above 1-day Donchian high (20-period) or below low with volume > 1.5x 20-period average
# Trend filter: 34-period EMA on daily timeframe to align with primary trend
# Works in bull/bear markets: breakouts capture trends, volatility filters reduce whipsaw
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Donchian20_1dEMA34_VolumeTrendFilter_v1"
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
    
    # Calculate daily Donchian channels and EMA ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA trend filter (34-period)
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above daily Donchian high with volume confirmation and uptrend
            if close[i] > donchian_high_aligned[i] and volume_filter[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below daily Donchian low with volume confirmation and downtrend
            elif close[i] < donchian_low_aligned[i] and volume_filter[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily Donchian low (failed breakout) or reaches opposite band (take profit)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily Donchian high (failed breakdown) or reaches opposite band (take profit)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals