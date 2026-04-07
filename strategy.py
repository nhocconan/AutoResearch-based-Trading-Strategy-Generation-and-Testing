#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Channel Breakout with 1d Trend Filter and Volume Confirmation
# Hypothesis: Donchian breakouts capture momentum in both bull and bear markets.
# In bull markets, upward breakouts capture strong trends; in bear markets, downward breakouts capture corrections.
# 1d EMA filter ensures trades align with higher timeframe trend, reducing whipsaw.
# Volume confirmation ensures institutional participation. 12h timeframe balances signal quality and trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
name = "12h_donchian20_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 12h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max()
    lower_channel = low_series.rolling(window=20, min_periods=20).min()
    middle_channel = (upper_channel + lower_channel) / 2
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches middle channel (take profit) or breaks below lower channel with volume
            if close[i] >= middle_channel[i] or (close[i] < lower_channel[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches middle channel (take profit) or breaks above upper channel with volume
            if close[i] <= middle_channel[i] or (close[i] > upper_channel[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price breaks above upper channel with trend confirmation
                if close[i] > upper_channel[i] and close[i] > daily_ema_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower channel with trend confirmation
                elif close[i] < lower_channel[i] and close[i] < daily_ema_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals