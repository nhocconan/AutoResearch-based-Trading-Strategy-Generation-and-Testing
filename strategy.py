#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high, weekly close > weekly open (bullish weekly candle), and volume > 1.5x 20-day average.
# Short when price breaks below 20-day low, weekly close < weekly open (bearish weekly candle), and volume > 1.5x 20-day average.
# Uses discrete position sizes (0.25) to minimize churn. Designed for 1d timeframe
# to capture major trend continuations while avoiding false breakouts in ranging markets.
# Target: 10-25 trades/year per symbol (~40-100 total over 4 years).
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_open = df_weekly['open'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly trend data to daily
    weekly_bullish = (weekly_close > weekly_open).astype(float)  # 1 for bullish, 0 for bearish
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian channels to be ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = high_max_20[i]
        lower_band = low_min_20[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        weekly_trend = weekly_trend_aligned[i]  # 1.0 for bullish weekly, 0.0 for bearish
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above 20-day high, weekly trend bullish, and volume confirmation
            if price > upper_band and weekly_trend > 0.5 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below 20-day low, weekly trend bearish, and volume confirmation
            elif price < lower_band and weekly_trend < 0.5 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below 20-day low or weekly trend turns bearish
            if price < lower_band or weekly_trend < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above 20-day high or weekly trend turns bullish
            if price > upper_band or weekly_trend > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals