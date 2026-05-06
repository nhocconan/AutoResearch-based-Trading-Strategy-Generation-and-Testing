#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and weekly EMA trend filter
# Long when price breaks above weekly Donchian upper channel (20-period high) and weekly EMA50 is rising
# Short when price breaks below weekly Donchian lower channel (20-period low) and weekly EMA50 is falling
# Uses weekly Donchian channels for key support/resistance, EMA for trend direction, volume for confirmation
# Designed to work in bull markets via breakouts above resistance and in bear markets via breakdowns below support
# Target: 10-20 trades per year (40-80 over 4 years) with 0.25 position sizing

name = "1d_weeklyDonchian20_EMA50_Trend_Volume"
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
    
    # Calculate weekly Donchian Channel (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period high and low for Donchian channels
    high_20 = df_1w['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1w, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Weekly EMA50 for trend filter
    ema50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    ema50_rising = ema50_aligned > np.roll(ema50_aligned, 1)
    ema50_rising[0] = False
    ema50_falling = ema50_aligned < np.roll(ema50_aligned, 1)
    ema50_falling[0] = False
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian with rising EMA50 and volume confirmation
            if close[i] > upper_donchian[i] and ema50_rising[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian with falling EMA50 and volume confirmation
            elif close[i] < lower_donchian[i] and ema50_falling[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian lower channel (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian upper channel (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals