#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA50 trend filter
# Uses 1d timeframe for signal generation with Donchian channel breakouts
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# 1w EMA50 > 1w EMA200 filters for bullish trend only, avoiding bearish whipsaws
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Donchian provides objective price channels, volume confirms breakout validity
# 1w EMA trend filter ensures trades only occur in favorable bullish conditions
# Works in bull markets by taking long trades; avoids bear markets by staying flat

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 1d
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Load 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 and EMA200
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMAs to 1d timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 1w bullish trend: EMA50 > EMA200
    bullish_trend = ema50_1w_aligned > ema200_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(bullish_trend[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Donchian upper + volume confirm + 1w bullish trend
            if close[i] > donchian_upper[i] and volume_confirm[i] and bullish_trend[i]:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Donchian lower (breakdown) or 1w bearish trend (EMA50 < EMA200)
            if close[i] < donchian_lower[i] or not bullish_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals