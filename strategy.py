#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades in bear markets
# Volume spike (>1.8x 20-period EMA) confirms institutional participation
# Target: 80-150 total trades over 4 years (20-38/year) to balance edge and fee drag
# Works in bull markets (mean reversion in uptrend) and bear markets (mean reversion in downtrend)

name = "4h_WilliamsR_MeanReversion_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 is overbought, < -80 is oversold
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Williams %R mean reversion signals with 1d trend filter
        # Long: Oversold (%R < -80) + price above 1d EMA50 + volume spike
        # Short: Overbought (%R > -20) + price below 1d EMA50 + volume spike
        if position == 0:
            if williams_r[i] < -80.0 and close[i] > ema_50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif williams_r[i] > -20.0 and close[i] < ema_50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses above 1d EMA50 (trend change) OR %R reaches overbought
            if close[i] < ema_50_1d_aligned[i] or williams_r[i] > -20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses below 1d EMA50 (trend change) OR %R reaches oversold
            if close[i] > ema_50_1d_aligned[i] or williams_r[i] < -80.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals