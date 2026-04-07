#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout + Daily EMA Filter + Volume Spike
# Hypothesis: Donchian breakouts on 12h timeframe, confirmed by daily EMA trend direction and volume spikes,
# capture sustained moves while avoiding false breakouts. Works in both bull and bear by following
# daily trend direction (long in uptrend, short in downtrend). Target: 20-30 trades/year (80-120 total).

name = "12h_donchian20_1d_ema_volume_v2"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA(30) for trend filter
    ema_30_1d = pd.Series(close_1d).ewm(span=30, adjust=False).mean().values
    ema_30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    
    # Donchian channels on 12h (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=10).max().values
    donchian_low = low_series.rolling(window=20, min_periods=10).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_30_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_30_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_30_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above Donchian high in uptrend
                if close[i] > donchian_high[i] and close[i] > ema_30_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below Donchian low in downtrend
                elif close[i] < donchian_low[i] and close[i] < ema_30_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals