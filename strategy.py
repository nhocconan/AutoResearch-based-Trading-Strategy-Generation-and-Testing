#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above Donchian(20) upper band AND 1w close > 1w EMA50 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below Donchian(20) lower band AND 1w close < 1w EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when price touches the opposite Donchian band (middle band) or reverses trend
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Donchian channels provide clear breakout levels with built-in stoploss via opposite band
# 1w EMA50 provides strong multi-timeframe trend filter to avoid counter-trend trades
# Volume spike (2.0x average) confirms institutional participation reducing false breakouts
# This combination worked well in experiments: 4h_Donchian20_1dEMA34_ATRVolume_v1 (Sharpe=0.233) and 4h_Camarilla_R3S3_1dEMA50_ATRVolume_v1 (Sharpe=0.125)

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
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
    
    # Calculate Donchian channels (20-period)
    # Upper band = highest high over past 20 bars
    # Lower band = lowest low over past 20 bars
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe (wait for completed HTF bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with trend and volume confirmation
            # Long breakout: price > upper band AND uptrend AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema50_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price < lower band AND downtrend AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema50_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches middle band OR trend reverses
            if close[i] <= donchian_middle[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches middle band OR trend reverses
            if close[i] >= donchian_middle[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals