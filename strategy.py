#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R extremes with 12h EMA50 trend filter and volume spike confirmation
# Long when 12h Williams %R < -80 (oversold) AND 12h EMA50 > EMA200 AND volume > 2.5 * avg_volume(24)
# Short when 12h Williams %R > -20 (overbought) AND 12h EMA50 < EMA200 AND volume > 2.5 * avg_volume(24)
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-150 total trades over 4 years (19-38/year) for 6h timeframe
# Williams %R identifies overextended moves; EMA filter ensures trend alignment; volume spike confirms conviction
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)

name = "6h_12hWilliamsR_Extreme_12hEMA50Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Williams %R and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 12h EMA50 and EMA200 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = close_series_12h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h indicators to 6h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate volume confirmation: volume > 2.5 * 24-period average volume
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.5 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with EMA50 > EMA200 and volume confirmation
            if (williams_r_aligned[i] < -80 and ema_50_aligned[i] > ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with EMA50 < EMA200 and volume confirmation
            elif (williams_r_aligned[i] > -20 and ema_50_aligned[i] < ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum fading)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum fading)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals