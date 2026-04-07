#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend and Volume Confirmation
# Hypothesis: Donchian breakouts capture momentum in both bull and bear markets.
# Weekly trend filter (EMA20 > SMA20) ensures we trade with the higher timeframe trend.
# Volume confirmation (volume > 1.5x 20-period average) filters for institutional participation.
# Target: 7-25 trades/year (30-100 total over 4 years) on 1d timeframe.
name = "1d_donchian_breakout_weekly_trend_volume_v1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    
    # Weekly trend filter: EMA20 > SMA20 = bullish
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_sma20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    weekly_ema20_1d = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    weekly_sma20_1d = align_htf_to_ltf(prices, df_1w, weekly_sma20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_ema20_1d[i]) or np.isnan(weekly_sma20_1d[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from weekly data
        bull_regime = weekly_ema20_1d[i] > weekly_sma20_1d[i]  # Bullish when EMA > SMA
        bear_regime = weekly_ema20_1d[i] < weekly_sma20_1d[i]  # Bearish when EMA < SMA
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band or trend reversal
            if close[i] <= donchian_lower[i] or not bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band or trend reversal
            if close[i] >= donchian_upper[i] or not bear_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Bull regime: look for long when price breaks above upper Donchian band
                if bull_regime and close[i] > donchian_upper[i]:
                    position = 1
                    signals[i] = 0.25
                # Bear regime: look for short when price breaks below lower Donchian band
                elif bear_regime and close[i] < donchian_lower[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals