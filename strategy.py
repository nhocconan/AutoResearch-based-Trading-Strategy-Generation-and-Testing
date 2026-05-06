#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout (20-week high/low) with volume confirmation and price above/below 50-week EMA trend filter
# Long when price breaks above weekly Donchian upper channel with volume > 1.5x 20-week average and price > 50-week EMA
# Short when price breaks below weekly Donchian lower channel with volume > 1.5x 20-week average and price < 50-week EMA
# Uses weekly Donchian channels for key support/resistance, volume for breakout confirmation, and EMA for trend filter
# Designed to work in bull markets via breakouts above resistance and in bear markets via breakdowns below target
# Target: 8-15 trades per year (32-60 over 4 years) with 0.30 position sizing

name = "1d_weeklyDonchian20_Volume_EMA50_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian Channel (20-week high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-week high and low for Donchian channels
    high_20w = df_1w['high'].rolling(window=20, min_periods=20).max().values
    low_20w = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1w, high_20w)
    lower_donchian = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Volume confirmation: >1.5x 20-week average volume
    vol_ma_20w = df_1w['volume'].rolling(window=20, min_periods=20).mean().values
    vol_ma_20w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20w)
    volume_filter = volume > (1.5 * vol_ma_20w_aligned)
    
    # 50-week EMA trend filter
    ema_50w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    trend_filter = close > ema_50w_aligned  # for long bias
    trend_filter_short = close < ema_50w_aligned  # for short bias
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if any critical value is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian upper with volume and trend confirmation
            if close[i] > upper_donchian[i] and volume_filter[i] and trend_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short breakdown: price breaks below weekly Donchian lower with volume and trend confirmation
            elif close[i] < lower_donchian[i] and volume_filter[i] and trend_filter_short[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian lower (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above weekly Donchian upper (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals