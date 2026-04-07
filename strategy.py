#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R + 1d EMA Trend Filter with Volume Confirmation
# Hypothesis: Williams %R (14) identifies overbought/oversold conditions on 6h.
# In bull markets (price > 1d EMA50), buy oversold pullbacks (%R < -80).
# In bear markets (price < 1d EMA50), sell overbought bounces (%R > -20).
# Volume filter ensures institutional participation. Designed for ranging/choppy markets
# where mean reversion at extremes works, but only in direction of higher timeframe trend.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "6h_williamsr_ema_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # 1d trend filter: price above/below 50 EMA
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Williams %R (14) on 6h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to overbought (> -20) or trend turns bearish or volume drops
            if (williams_r[i] > -20 or close[i] < ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: Williams %R returns to oversold (< -80) or trend turns bullish or volume drops
            if (williams_r[i] < -80 or close[i] > ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: Williams %R oversold (< -80) with bullish trend and volume
            if (williams_r[i] < -80 and close[i] > ema_50[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R overbought (> -20) with bearish trend and volume
            elif (williams_r[i] > -20 and close[i] < ema_50[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals