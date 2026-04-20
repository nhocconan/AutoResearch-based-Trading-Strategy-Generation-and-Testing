#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation
# - Donchian breakout on 12h provides clear entry/exit signals with low trade frequency
# - Weekly EMA(21) trend filter ensures trades align with long-term direction
# - Volume spike confirmation filters false breakouts (volume > 1.5x 20-period average)
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(21) on weekly timeframe
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Donchian channels on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Upper and lower bands (20-period)
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = vol_avg * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_threshold[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        volume = volume_12h[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        trend = ema_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above upper band + volume confirmation + uptrend
            if price > upper and volume > volume_threshold[i] and price > trend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower band + volume confirmation + downtrend
            elif price < lower and volume > volume_threshold[i] and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below lower band (reversal) or trend turns bearish
            if price < lower or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above upper band (reversal) or trend turns bullish
            if price > upper or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0