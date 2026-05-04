#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets, it provides mean reversion entries.
# In trending markets, the 1d EMA50 filter ensures we only take trades in the direction of the higher timeframe trend.
# Volume confirmation (1.5x average) ensures strong participation, reducing false signals.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag on 4h timeframe.
# Works in both bull and bear markets by adapting to the 1d trend direction.
# Prioritizes BTC/ETH performance.

name = "4h_WilliamsR_MeanReversion_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values below -80 = oversold, above -20 = overbought
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-9) * -100
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams %R mean reversion with 1d trend filter
        # Long: Oversold (%R < -80) + volume spike + price above 1d EMA50 (uptrend)
        # Short: Overbought (%R > -20) + volume spike + price below 1d EMA50 (downtrend)
        if position == 0:
            if (williams_r[i] < -80 and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (williams_r[i] > -20 and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 1d EMA50 (trend change) OR Williams %R becomes overbought
            if close[i] < ema_50_1d_aligned[i] or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 1d EMA50 (trend change) OR Williams %R becomes oversold
            if close[i] > ema_50_1d_aligned[i] or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals