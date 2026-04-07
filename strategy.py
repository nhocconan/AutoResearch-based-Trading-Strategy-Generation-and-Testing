#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout + 1d EMA Trend + Volume Confirmation
# Hypothesis: Donchian(20) breakout on 12h identifies breakouts with momentum.
# Trend filter: 1d EMA50 > EMA200 for bullish bias, EMA50 < EMA200 for bearish bias.
# Volume confirmation: current volume > 1.5x 20-period average reduces false breakouts.
# In bull trend: long on upper band breakout with volume.
# In bear trend: short on lower band breakout with volume.
# Uses 12h timeframe for lower frequency (~15-35 trades/year) to minimize fee drag.
# Risk management: exit when price crosses opposite Donchian band or trend reverses.
name = "12h_donchian20_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 12-period EMA for Donchian calculation (not needed, using pure high/low)
    # Donchian Channel (20-period) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA50 and EMA200 for trend filter
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema200 = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_ema50_12h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    daily_ema200_12h = align_htf_to_ltf(prices, df_1d, daily_ema200)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(daily_ema50_12h[i]) or np.isnan(daily_ema200_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from 1d data
        bull_trend = daily_ema50_12h[i] > daily_ema200_12h[i]  # Bullish when EMA50 > EMA200
        bear_trend = daily_ema50_12h[i] < daily_ema200_12h[i]  # Bearish when EMA50 < EMA200
        
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian band or trend turns bearish
            if close[i] <= donchian_low[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian band or trend turns bullish
            if close[i] >= donchian_high[i] or not bear_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Bull trend: look for long on upper band breakout
                if bull_trend and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Bear trend: look for short on lower band breakout
                elif bear_trend and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals