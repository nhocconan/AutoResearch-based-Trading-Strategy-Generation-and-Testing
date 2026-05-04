#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA34 trend filter and volume spike confirmation
# Uses 1w HTF for strong trend alignment to avoid whipsaw in ranging markets.
# Camarilla R4/S4 from 1d provide institutional support/resistance levels with wider bands than R3/S3.
# Volume confirmation (2.5x 20-period EMA) ensures breakout conviction.
# Designed for 1d timeframe targeting 15-25 trades/year (60-100 total) with discrete sizing (0.28).
# Works in bull markets by buying R4 breakouts in uptrends and bear markets by selling S4 breakdowns in downtrends.
# The 1w EMA34 trend filter provides strong trend detection with minimal lag.

name = "1d_Camarilla_R4S4_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (R4, S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate camarilla levels: R4, S4 from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r4 = close_1d + 1.1 * camarilla_range
    s4 = close_1d - 1.1 * camarilla_range
    
    # Align camarilla levels to 1d timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 2.5x 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.5 x 20-period EMA
        volume_confirmed = volume[i] > (2.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: close breaks above R4 + volume confirmation + price above 1w EMA34 (uptrend)
            if (close[i] > r4_aligned[i] and volume_confirmed and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.28
                position = 1
            # Short: close breaks below S4 + volume confirmation + price below 1w EMA34 (downtrend)
            elif (close[i] < s4_aligned[i] and volume_confirmed and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price falls below S4 (mean reversion) OR below 1w EMA34 (trend change)
            if close[i] < s4_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price rises above R4 (mean reversion) OR above 1w EMA34 (trend change)
            if close[i] > r4_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals