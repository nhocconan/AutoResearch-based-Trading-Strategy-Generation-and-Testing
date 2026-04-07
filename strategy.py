#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Williams %R + 1w Trend + Volume Confirmation
# Hypothesis: Williams %R identifies overbought/oversold conditions on daily timeframe.
# In oversold (%R < -80) with weekly uptrend, go long. In overbought (%R > -20) with weekly downtrend, go short.
# Volume confirms institutional participation. Williams %R is effective in both ranging and trending markets.
# 1d timeframe balances responsiveness and noise reduction. Target: 7-25 trades/year (30-100 over 4 years).
name = "1d_williams_r_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Williams %R on 1d timeframe (14 period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1-week EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(weekly_ema_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (momentum fading) or weekly trend turns down
            if williams_r[i] > -50 or close[i] < weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (momentum fading) or weekly trend turns up
            if williams_r[i] < -50 or close[i] > weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Oversold with weekly uptrend: go long
                if williams_r[i] < -80 and close[i] > weekly_ema_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Overbought with weekly downtrend: go short
                elif williams_r[i] > -20 and close[i] < weekly_ema_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals