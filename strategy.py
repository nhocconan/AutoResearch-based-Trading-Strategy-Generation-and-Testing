#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout + Weekly Trend + Volume Confirmation
# Hypothesis: Donchian(20) breakouts on daily chart with weekly trend filter and volume
# confirmation captures strong trends while avoiding false breakouts. Works in bull via
# upper band breakouts, in bear via lower band breakdowns. Target: 10-25 trades/year.
name = "1d_donchian20_weekly_trend_volume_v1"
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
    
    # Get daily data for Donchian channels (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels on daily data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    donchian_upper = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (no shift needed as we're already on 1d)
    # Since we're using daily data on daily timeframe, alignment is direct
    # But we still use the helper for consistency and proper handling of gaps
    donchian_upper_1d = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_1d = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: daily volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper_1d[i]) or np.isnan(donchian_lower_1d[i]) or
            np.isnan(weekly_ema_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or weekly trend turns bearish
            if close[i] < donchian_lower_1d[i] or close[i] < weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or weekly trend turns bullish
            if close[i] > donchian_upper_1d[i] or close[i] > weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian upper with volume and bullish weekly trend
            if close[i] > donchian_upper_1d[i] and vol_confirm and close[i] > weekly_ema_1d[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower with volume and bearish weekly trend
            elif close[i] < donchian_lower_1d[i] and vol_confirm and close[i] < weekly_ema_1d[i]:
                position = -1
                signals[i] = -0.25
    
    return signals