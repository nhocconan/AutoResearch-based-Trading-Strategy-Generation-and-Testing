#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and volatility filter
# Long when price breaks above weekly Donchian upper channel (20-period high) with volume > 1.5x 20-period average and ATR ratio < 1.2
# Short when price breaks below weekly Donchian lower channel (20-period low) with volume > 1.5x 20-period average and ATR ratio < 1.2
# Uses weekly Donchian channels for key support/resistance levels, volume for confirmation, and ATR ratio to avoid high volatility periods
# Designed to work in bull markets via breakouts above resistance and in bear markets via breakdowns below support
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing

name = "1d_weeklyDonchian20_Volume_VolatilityFilter_v1"
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
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Volatility filter: ATR ratio (current ATR / 20-period average ATR) < 1.2 to avoid high volatility periods
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma
    volatility_filter = (atr_ratio < 1.2) & ~np.isnan(atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any critical value is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian upper with volume and volatility filters
            if close[i] > upper_donchian[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian lower with volume and volatility filters
            elif close[i] < lower_donchian[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian lower (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian upper (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals