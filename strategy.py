#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions. In a trending market (price > 1d EMA50 for longs,
# price < 1d EMA50 for shorts), extreme Williams %R readings (< -80 for longs, > -20 for shorts) 
# with volume confirmation (2.0x 20-period EMA) provide high-probability mean reversion entries.
# Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Uses 1d EMA50 for smoother trend filter that avoids whipsaws in ranging markets while capturing major trends.
# Works in bull markets by buying oversold pullbacks in uptrends and in bear markets by selling overbought 
# rallies in downtrends, avoiding false signals via trend and volume filters.

name = "6h_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume confirmation: 2.0x 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + volume confirmation + price above 1d EMA50 (uptrend)
            if (williams_r[i] < -80.0 and volume_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + volume confirmation + price below 1d EMA50 (downtrend)
            elif (williams_r[i] > -20.0 and volume_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (return from oversold) OR below 1d EMA50 (trend change)
            if williams_r[i] > -50.0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (return from overbought) OR above 1d EMA50 (trend change)
            if williams_r[i] < -50.0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals