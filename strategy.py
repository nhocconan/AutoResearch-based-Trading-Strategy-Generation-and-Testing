#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with volume > 1.5x average and 1d uptrend (EMA50).
# Enter short when price breaks below Donchian(20) low with volume > 1.5x average and 1d downtrend.
# Exit when price crosses the Donchian midpoint (mean reversion) or volume drops below average.
# Uses Donchian channel for breakout, 1d EMA for trend filter, volume for confirmation.
# Designed for fewer trades (~20-40/year) to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    
    # Calculate Donchian channel (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate 1d trend filter: EMA50
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_4h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or \
           np.isnan(daily_ema50_4h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midpoint (mean reversion) or volume drops below average
            if close[i] <= donch_mid[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midpoint (mean reversion) or volume drops below average
            if close[i] >= donch_mid[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema50_4h[i]
            daily_downtrend = close[i] < daily_ema50_4h[i]
            
            # Long entry: price breaks above Donchian high with volume and uptrend
            if close[i] > donch_high[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and downtrend
            elif close[i] < donch_low[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals