#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian breakout + volume + 1d trend filter
# Hypothesis: Buy breakout above 20-period Donchian high in uptrend, sell breakdown below 20-period low in downtrend.
# Uses 1-day EMA(50) for trend filter and volume confirmation to avoid false breakouts.
# Works in bull by buying breakouts, works in bear by selling breakdowns.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA(50) for trend filter
    close_daily = df_daily['close'].values
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend changes to downtrend
            if close[i] < low_roll[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend changes to uptrend
            if close[i] > high_roll[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of daily trend with volume confirmation
            if vol_ok:
                if close[i] > ema_50_4h[i]:  # Uptrend
                    if close[i] > high_roll[i]:  # Breakout above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if close[i] < low_roll[i]:  # Breakdown below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals