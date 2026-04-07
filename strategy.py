#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Donchian Breakout + 4h Trend + Volume Spike
# Hypothesis: 1h breakouts in direction of 4h trend with volume confirmation capture
# strong moves in both bull and bear markets. 4h trend filter reduces false breakouts.
# Volume spike confirms institutional participation. Designed for low trade frequency.
# Works in bull via long breakouts in uptrend, in bear via short breakouts in downtrend.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_donchian_breakout_4h_trend_volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 8-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low OR 4h trend turns bearish
            if close[i] < low_20[i] or close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high OR 4h trend turns bullish
            if close[i] > high_20[i] or close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above 20-period high with 4h uptrend
                if close[i] > high_20[i] and close[i] > ema_20_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below 20-period low with 4h downtrend
                elif close[i] < low_20[i] and close[i] < ema_20_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals