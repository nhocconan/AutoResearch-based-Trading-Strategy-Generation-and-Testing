#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 12h Trend + Volume Spike
# Hypothesis: Donchian(20) breakouts on 4h capture trend continuations. 
# Trend filter from 12h EMA(50) ensures alignment with higher timeframe direction.
# Volume spike (>2x 20-period average) confirms institutional participation and reduces false breakouts.
# Works in bull markets (breakouts continue uptrend) and bear markets (breakouts continue downtrend).
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.
name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 4h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max()
    lower_channel = low_series.rolling(window=20, min_periods=20).min()
    
    # 12-hour EMA(50) for trend filter
    daily_close = df_12h['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_12h, daily_ema)
    
    # Volume filter: current volume > 2x 20-period average (reduces trade frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(daily_ema_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches opposite Donchian band (mean reversion exit)
            if close[i] <= lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches opposite Donchian band (mean reversion exit)
            if close[i] >= upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Maintain short position
        else:  # Flat, look for entry
            # Require volume spike for confirmation
            if vol_spike[i]:
                # Long: price breaks above upper Donchian band with trend alignment
                if close[i] > upper_channel[i] and close[i] > daily_ema_4h[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: price breaks below lower Donchian band with trend alignment
                elif close[i] < lower_channel[i] and close[i] < daily_ema_4h[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals