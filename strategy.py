#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Daily Volume & Weekly Trend
# Hypothesis: 12h Donchian(20) breakouts capture medium-term trends.
# Daily volume > 2.0x average confirms institutional participation.
# Weekly EMA(50) filter ensures alignment with longer-term trend.
# Works in bull: breaks above upper band with volume + above weekly EMA.
# Works in bear: breaks below lower band with volume + below weekly EMA.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_donchian20_volume_weekly_trend_v1"
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily volume confirmation: volume > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Weekly trend filter: EMA(50) of weekly close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or weekly trend turns bearish
            if close[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or weekly trend turns bullish
            if close[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long breakout: price closes above upper band with weekly uptrend
                if close[i] > donchian_upper[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below lower band with weekly downtrend
                elif close[i] < donchian_lower[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals