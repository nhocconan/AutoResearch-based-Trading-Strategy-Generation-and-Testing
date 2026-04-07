#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend and Volume Confirmation
# Hypothesis: Daily Donchian channel breakouts (20-period) combined with weekly trend filter
# and volume confirmation captures institutional moves across market cycles. Works in bull via
# upper band breakouts, bear via lower band breakdowns, and avoids whipsaws in ranging markets.
# Target: 15-25 trades/year to minimize fee drag.
name = "daily_donchian20_weekly_trend_volume_v2"
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
    
    # Calculate Donchian channels (20-period high/low) on daily data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # 20-period rolling max/min for Donchian channels
    upper_channel = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (no shift needed as already aligned)
    upper_daily = upper_channel
    lower_daily = lower_channel
    
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
        if (np.isnan(upper_daily[i]) or np.isnan(lower_daily[i]) or 
            np.isnan(weekly_ema_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian channel or weekly trend turns bearish
            if close[i] < lower_daily[i] or close[i] < weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian channel or weekly trend turns bullish
            if close[i] > upper_daily[i] or close[i] > weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above upper Donchian channel with volume and bullish weekly trend
            if close[i] > upper_daily[i] and vol_confirm and close[i] > weekly_ema_1d[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below lower Donchian channel with volume and bearish weekly trend
            elif close[i] < lower_daily[i] and vol_confirm and close[i] < weekly_ema_1d[i]:
                position = -1
                signals[i] = -0.25
    
    return signals