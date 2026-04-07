#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout + 1W EMA Trend + Volume Confirmation
# Hypothesis: Donchian breakouts capture strong directional moves on 12h timeframe.
# We filter trades by weekly EMA(50) trend direction to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction. This structure works in both bull
# and bear markets because we only trade in the direction of the higher-timeframe trend.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_donchian20_1w_ema_volume_v1"
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
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 12h timeframe
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1-week EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    weekly_ema_12h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_ema_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend turns bearish
            if close[i] < donchian_low[i] or close[i] < weekly_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend turns bullish
            if close[i] > donchian_high[i] or close[i] > weekly_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price breaks above Donchian upper band + above weekly EMA
                if close[i] > donchian_high[i] and close[i] > weekly_ema_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band + below weekly EMA
                elif close[i] < donchian_low[i] and close[i] < weekly_ema_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals