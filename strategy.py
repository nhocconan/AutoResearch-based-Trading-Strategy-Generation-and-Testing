#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 12h EMA trend + volume confirmation.
# Williams %R identifies overbought/oversold conditions; mean reversion in ranging markets.
# In trending markets (12h EMA slope), fade extreme %R readings for counter-trend entries.
# Long when %R < -80 (oversold) and 12h EMA uptrend; short when %R > -20 (overbought) and 12h EMA downtrend.
# Volume confirmation (>1.5x 20-period average) filters low-quality signals.
# Exit when %R reverts to -50 (mean) or trend weakens.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 25-35 trades/year per symbol (100-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R parameters
    willr_period = 14
    
    # Load 12h data ONCE for Williams %R and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < willr_period:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=willr_period, min_periods=willr_period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=willr_period, min_periods=willr_period).min().values
    ws = highest_high - lowest_low
    # Avoid division by zero
    ws = np.where(ws == 0, 1e-10, ws)
    willr_12h = ((highest_high - close_12h) / ws) * -100
    
    # Calculate EMA(21) on 12h for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    # EMA slope: current EMA - previous EMA
    ema_slope = np.diff(ema_12h, prepend=np.nan)
    
    # Align indicators to lower timeframe
    willr_12h_aligned = align_htf_to_ltf(prices, df_12h, willr_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(willr_period, 21, 20)  # Need Williams %R, EMA, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(willr_12h_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: EMA slope positive for uptrend, negative for downtrend
        uptrend = ema_slope_aligned[i] > 0
        downtrend = ema_slope_aligned[i] < 0
        
        if position == 0:
            # Look for mean reversion entries
            # Long: Williams %R oversold (< -80) AND uptrend
            if (willr_12h_aligned[i] < -80 and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) AND downtrend
            elif (willr_12h_aligned[i] > -20 and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R reverts to mean (>= -50) or trend weakens (EMA slope <= 0)
            if (willr_12h_aligned[i] >= -50 or 
                ema_slope_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R reverts to mean (<= -50) or trend weakens (EMA slope >= 0)
            if (willr_12h_aligned[i] <= -50 or 
                ema_slope_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_12hEMA_Trend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0