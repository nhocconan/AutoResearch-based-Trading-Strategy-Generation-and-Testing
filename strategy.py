#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Daily Trend + Volume Confirmation
# Hypothesis: Donchian(20) breakouts on 4h combined with daily trend filter and volume
# confirmation captures institutional breakouts. Works in bull via upper band breaks,
# in bear via lower band breaks, and ranges via mean reversion at mid-band.
# Target: 20-40 trades/year to minimize fee drag.
name = "4h_donchian20_daily_trend_volume_v1"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Donchian channels (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: 4h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(daily_ema_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian mid (mean reversion) or daily trend turns bearish
            if close[i] < donchian_mid[i] or close[i] < daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian mid (mean reversion) or daily trend turns bullish
            if close[i] > donchian_mid[i] or close[i] > daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian high (breakout) with volume and bullish daily trend
            if close[i] > donchian_high[i] and vol_confirm and close[i] > daily_ema_4h[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low (breakdown) with volume and bearish daily trend
            elif close[i] < donchian_low[i] and vol_confirm and close[i] < daily_ema_4h[i]:
                position = -1
                signals[i] = -0.25
            # Enter long: price crosses above Donchian mid from below (mean reversion long) with volume
            elif close[i] > donchian_mid[i] and close[i-1] <= donchian_mid[i-1] and vol_confirm:
                position = 1
                signals[i] = 0.20
            # Enter short: price crosses below Donchian mid from above (mean reversion short) with volume
            elif close[i] < donchian_mid[i] and close[i-1] >= donchian_mid[i-1] and vol_confirm:
                position = -1
                signals[i] = -0.20
    
    return signals