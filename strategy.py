#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels (R3/S3) from weekly data act as strong support/resistance. A breakout
# above R3 or below S3 with volume confirmation (2.0x 50-period EMA) and alignment with
# 1w EMA50 trend provides high-probability continuation entries. Designed for 12h timeframe
# to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Works in bull markets by buying breakouts above R3 in uptrends and in bear markets by selling
# breakdowns below S3 in downtrends, avoiding false breakouts via trend and volume filters.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1w data for EMA50 trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate camarilla levels: R3, S3 from 1w OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_range = high_1w - low_1w
    r3 = close_1w + 1.1 * camarilla_range / 2
    s3 = close_1w - 1.1 * camarilla_range / 2
    
    # Align camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: 2.0x 50-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_50 = vol_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 50-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_50[i])
        
        if position == 0:
            # Long: close breaks above R3 + volume confirmation + price above 1w EMA50 (uptrend)
            if (close[i] > r3_aligned[i] and volume_confirmed and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S3 + volume confirmation + price below 1w EMA50 (downtrend)
            elif (close[i] < s3_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 (mean reversion) OR below 1w EMA50 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 (mean reversion) OR above 1w EMA50 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals