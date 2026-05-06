#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and volatility filter
# Weekly Donchian channels (20-period high/low) identify strong breakouts with follow-through
# Volume confirmation (>1.5x 20-day average) ensures institutional participation
# Volatility filter (ATR ratio < 0.8) avoids choppy markets and false breakouts
# Works in bull/bear markets: breakouts capture trends, volatility filter avoids whipsaws
# Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_WeeklyDonchian20_VolumeVolFilter_v1"
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
    
    # Calculate weekly Donchian channels ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    high_series = pd.Series(df_1w['high'])
    low_series = pd.Series(df_1w['low'])
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Volatility filter: ATR ratio < 0.8 (avoid choppy markets)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr < (0.8 * atr_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume and volatility filter
            if close[i] > donchian_high_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly Donchian low with volume and volatility filter
            elif close[i] < donchian_low_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (failed breakout) or trailing stop
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (failed breakdown) or trailing stop
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals