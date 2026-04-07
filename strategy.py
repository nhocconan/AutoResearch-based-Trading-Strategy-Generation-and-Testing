#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian(20) breakout + 1w EMA trend + volume confirmation
# Hypothesis: Buy breakouts above 20-day high in uptrend, sell breakdowns below 20-day low in downtrend.
# Volume confirms breakout strength. Works in bull/bear by following 1-week trend.
# Target: 20-80 trades over 4 years (5-20/year).

name = "1d_donchian20_1w_ema_volume_v1"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Donchian channels (20-period)
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # 20-period SMA for volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend changes
            if close[i] < lower[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend changes
            if close[i] > upper[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_confirm:
                # Breakout above upper band with uptrend
                if close[i] > upper[i] and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower band with downtrend
                elif close[i] < lower[i] and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals