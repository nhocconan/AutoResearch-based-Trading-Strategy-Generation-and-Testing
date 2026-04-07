#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R + 1d Trend + Volume Confirmation
# Hypothesis: Williams %R identifies overbought/oversold conditions on 6h timeframe.
# In ranging markets (Williams %R between -20 and -80), we mean-revert at extremes.
# In trending markets (confirmed by 1d EMA), we follow breakouts when Williams %R exits extremes.
# Volume confirms institutional participation. This adapts to both ranging and trending markets.
# 6h timeframe balances responsiveness and noise reduction. Target: 12-37 trades/year (50-150 over 4 years).
name = "6h_williams_r_1d_trend_volume_v1"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Williams %R on 6h timeframe (14 periods)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(daily_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns above -50 (mean reversion) or breaks below -80 with volume (stop)
            if williams_r[i] > -50 or (williams_r[i] < -80 and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (mean reversion) or breaks above -20 with volume (stop)
            if williams_r[i] < -50 or (williams_r[i] > -20 and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Ranging market: Williams %R between -80 and -20, mean revert at extremes
                if -80 <= williams_r[i] <= -20:
                    # Long: Williams %R crosses above -80 from below (oversold bounce)
                    if williams_r[i] > -80 and williams_r[i-1] <= -80:
                        position = 1
                        signals[i] = 0.25
                    # Short: Williams %R crosses below -20 from above (overbought rejection)
                    elif williams_r[i] < -20 and williams_r[i-1] >= -20:
                        position = -1
                        signals[i] = -0.25
                # Trending market: Williams %R outside normal range, follow momentum with trend filter
                else:
                    # Strong oversold (< -80) with uptrend: potential breakout long
                    if williams_r[i] < -80 and close[i] > daily_ema_6h[i]:
                        position = 1
                        signals[i] = 0.25
                    # Strong overbought (> -20) with downtrend: potential breakdown short
                    elif williams_r[i] > -20 and close[i] < daily_ema_6h[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals