#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band with price > 1d EMA50 and volume > 1.5x average
# Short when price breaks below 4h Donchian lower band with price < 1d EMA50 and volume > 1.5x average
# Uses 4h Donchian for price channels, 1d EMA50 for trend filter, volume for confirmation
# Designed to work in bull markets via breakouts above resistance and in bear markets via breakdowns below support
# Target: 15-37 trades per year (60-150 over 4 years) with 0.20 position sizing

name = "1h_4hDonchian_1dEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h Donchian channels (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    donchian_upper = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA50
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 4h Donchian upper with uptrend and volume confirmation
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short breakout: price breaks below 4h Donchian lower with downtrend and volume confirmation
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian lower (support break)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h Donchian upper (resistance break)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals